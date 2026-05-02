//! Procedural macros for ember-rs.

extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::{abort_call_site, proc_macro_error};
use quote::{format_ident, quote};
use std::fs;
use structmeta::StructMeta;
use syn::{parse_macro_input, ItemStruct, LitStr};

use tflite_flatbuffers::tflite::{
    root_as_model, ActivationFunctionType, Buffer, BuiltinOperator, Padding, Tensor, TensorType,
};

#[path = "../flatbuffers/tflite_generated.rs"]
#[allow(unused_imports)]
#[allow(clippy::all)]
mod tflite_flatbuffers;

#[derive(StructMeta)]
struct Args {
    #[struct_meta(unnamed)]
    path: LitStr,
}

#[derive(Clone)]
struct TensorInfo {
    shape: Vec<usize>,
    scale: f32,
    zero_point: i32,
    tensor_type: TensorType,
}

/// Generate a backend-dispatched inference wrapper from a quantized TFLite model.
///
/// The generated impl exposes:
///
/// - `input_len() -> usize`
/// - `output_len() -> usize`
/// - `predict_quantized<B: ember_core::KernelBackend>(...) -> ember_core::Status`
///
/// The backend is supplied by the caller, so model code can switch between
/// `ember-ref`, `ember-esp`, or any custom backend implementing
/// `ember_core::KernelBackend`.
#[proc_macro_error]
#[proc_macro_attribute]
pub fn model(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as Args);
    let item = parse_macro_input!(item as ItemStruct);

    let buf = fs::read(args.path.value()).unwrap_or_else(|_| {
        abort_call_site!(
            "couldn't find '{}', please provide a valid path",
            &args.path.value()
        )
    });
    let model = root_as_model(&buf).unwrap_or_else(|_| {
        abort_call_site!("invalid model, please provide a valid TensorFlow Lite model")
    });

    let ident = &item.ident;
    let subgraph = model.subgraphs().unwrap().get(0);
    let tensors = subgraph.tensors().unwrap();
    let buffers = model.buffers().unwrap();
    let operators = subgraph.operators().unwrap();

    let input_tensor_id = subgraph.inputs().unwrap().get(0) as usize;
    let output_tensor_id = subgraph.outputs().unwrap().get(0) as usize;
    let input_info = tensor_info(tensors.get(input_tensor_id));
    let output_info = tensor_info(tensors.get(output_tensor_id));

    require_i8(&input_info, "model input");
    require_i8(&output_info, "model output");

    let input_len = tensor_len(&input_info);
    let output_len = tensor_len(&output_info);

    let mut body = TokenStream2::new();
    let mut scratch_queries = TokenStream2::new();

    for (index, operator) in operators.iter().enumerate() {
        let op_code_table = model
            .operator_codes()
            .unwrap()
            .get(operator.opcode_index() as usize);

        let opcode = BuiltinOperator(if op_code_table.deprecated_builtin_code() as i32 == 127 {
            op_code_table.builtin_code().0
        } else {
            op_code_table.deprecated_builtin_code() as i32
        });

        let layer = match opcode {
            BuiltinOperator::CONV_2D => emit_conv2d(
                index,
                operator,
                &tensors,
                &buffers,
                input_tensor_id,
                output_tensor_id,
            ),
            BuiltinOperator::DEPTHWISE_CONV_2D => emit_depthwise_conv2d(
                index,
                operator,
                &tensors,
                &buffers,
                input_tensor_id,
                output_tensor_id,
            ),
            BuiltinOperator::FULLY_CONNECTED => emit_fully_connected(
                index,
                operator,
                &tensors,
                &buffers,
                input_tensor_id,
                output_tensor_id,
            ),
            BuiltinOperator::AVERAGE_POOL_2D => emit_pool(
                index,
                operator,
                &tensors,
                input_tensor_id,
                output_tensor_id,
                true,
            ),
            BuiltinOperator::MAX_POOL_2D => emit_pool(
                index,
                operator,
                &tensors,
                input_tensor_id,
                output_tensor_id,
                false,
            ),
            BuiltinOperator::SOFTMAX => {
                emit_softmax(index, operator, &tensors, input_tensor_id, output_tensor_id)
            }
            BuiltinOperator::ADD => emit_add(operator, &tensors, input_tensor_id, output_tensor_id),
            BuiltinOperator::RESHAPE => {
                emit_reshape(operator, &tensors, input_tensor_id, output_tensor_id)
            }
            unsupported_op => abort_call_site!("unsupported operator: {:?}", unsupported_op),
        };

        body.extend(layer);
        scratch_queries.extend(emit_scratch_query(opcode, operator, &tensors));
    }

    let ts = quote! {
        #item

        impl #ident {
            /// Number of quantized input elements expected by this model.
            pub const fn input_len() -> usize {
                #input_len
            }

            /// Number of quantized output elements written by this model.
            pub const fn output_len() -> usize {
                #output_len
            }

            /// Maximum scratch size in bytes required by this model for backend `B`.
            pub fn scratch_len<B: ember_core::KernelBackend>() -> usize {
                let mut required_scratch = 0usize;
                #scratch_queries
                required_scratch
            }

            /// Run INT8 inference with a caller-supplied backend.
            ///
            /// Pass any backend that implements [`ember_core::KernelBackend`],
            /// for example `ember_ref::RefBackend`, `ember_esp::EspBackend`, or
            /// a custom backend from your own crate.
            pub fn predict_quantized<B: ember_core::KernelBackend>(
                backend: &mut B,
                input: &[i8],
                output: &mut [i8],
            ) -> ember_core::Status {
                let mut scratch = [];
                Self::predict_quantized_with_scratch(backend, input, output, &mut scratch)
            }

            /// Run INT8 inference with a caller-supplied backend and scratch buffer.
            ///
            /// Use this entry point for optimized backends that need temporary
            /// memory. Allocate at least `Self::scratch_len::<B>()` bytes.
            pub fn predict_quantized_with_scratch<B: ember_core::KernelBackend>(
                backend: &mut B,
                input: &[i8],
                output: &mut [i8],
                scratch: &mut [u8],
            ) -> ember_core::Status {
                if input.len() != Self::input_len() || output.len() != Self::output_len() {
                    return Err(ember_core::KernelError::InvalidShape);
                }

                debug_assert!(
                    scratch.len() >= Self::scratch_len::<B>(),
                    "ember: scratch buffer too small - need {} bytes, got {}. \
                     Allocate at least Self::scratch_len::<B>() bytes and use \
                     predict_quantized_with_scratch.",
                    Self::scratch_len::<B>(),
                    scratch.len()
                );

                #body

                Ok(())
            }
        }
    };

    fs::create_dir_all("target").ok();
    fs::write("target/ember-expansion.rs", ts.to_string()).ok();

    ts.into()
}

fn tensor_info(tensor: Tensor<'_>) -> TensorInfo {
    let mut shape: Vec<_> = tensor
        .shape()
        .unwrap()
        .iter()
        .map(|dim| dim as usize)
        .collect();
    if shape.len() == 1 {
        shape.insert(0, 1);
    }

    let quant = tensor.quantization().unwrap();
    let scale = quant.scale().and_then(|s| s.iter().next()).unwrap_or(1.0);
    let zero_point = quant
        .zero_point()
        .and_then(|z| z.iter().next())
        .unwrap_or(0) as i32;

    TensorInfo {
        shape,
        scale,
        zero_point,
        tensor_type: tensor.type_(),
    }
}

fn tensor_len(info: &TensorInfo) -> usize {
    info.shape.iter().product()
}

fn require_i8(info: &TensorInfo, name: &str) {
    if info.tensor_type != TensorType::INT8 {
        abort_call_site!(
            "{} must be INT8 for ember backend dispatch, got {:?}",
            name,
            info.tensor_type
        );
    }
}

fn buffer_data(buffer: Buffer<'_>) -> Vec<u8> {
    buffer
        .data()
        .map(|data| data.iter().collect())
        .unwrap_or_default()
}

fn tensor_i8_data(
    tensor: Tensor<'_>,
    buffers: &flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<Buffer<'_>>>,
) -> Vec<i8> {
    if tensor.type_() != TensorType::INT8 {
        abort_call_site!("constant tensor must be INT8, got {:?}", tensor.type_());
    }

    buffer_data(buffers.get(tensor.buffer() as usize))
        .into_iter()
        .map(|byte| byte as i8)
        .collect()
}

fn tensor_i32_data(
    tensor: Tensor<'_>,
    buffers: &flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<Buffer<'_>>>,
) -> Vec<i32> {
    buffer_data(buffers.get(tensor.buffer() as usize))
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn shape2(info: &TensorInfo, name: &str) -> [usize; 2] {
    if info.shape.len() != 2 {
        abort_call_site!("{} must be rank 2, got shape {:?}", name, info.shape);
    }
    [info.shape[0], info.shape[1]]
}

fn shape4(info: &TensorInfo, name: &str) -> [usize; 4] {
    if info.shape.len() != 4 {
        abort_call_site!("{} must be rank 4, got shape {:?}", name, info.shape);
    }
    [info.shape[0], info.shape[1], info.shape[2], info.shape[3]]
}

fn shape2_tokens(shape: [usize; 2]) -> TokenStream2 {
    let [a, b] = shape;
    quote!([#a, #b])
}

fn shape4_tokens(shape: [usize; 4]) -> TokenStream2 {
    let [a, b, c, d] = shape;
    quote!([#a, #b, #c, #d])
}

fn quant_tokens(info: &TensorInfo) -> TokenStream2 {
    let scale = info.scale;
    let zero_point = info.zero_point;
    quote!(ember_core::QuantParam {
        scale: #scale,
        zero_point: #zero_point,
    })
}

fn emit_scratch_query(
    opcode: BuiltinOperator,
    operator: tflite_flatbuffers::tflite::Operator<'_>,
    tensors: &flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<Tensor<'_>>>,
) -> TokenStream2 {
    match opcode {
        BuiltinOperator::CONV_2D => {
            let inputs = operator.inputs().unwrap();
            let input_shape = shape4_tokens(shape4(
                &tensor_info(tensors.get(inputs.get(0) as usize)),
                "Conv2D input",
            ));
            let weights_shape = shape4_tokens(shape4(
                &tensor_info(tensors.get(inputs.get(1) as usize)),
                "Conv2D weights",
            ));
            let output_shape = shape4_tokens(shape4(
                &tensor_info(tensors.get(operator.outputs().unwrap().get(0) as usize)),
                "Conv2D output",
            ));
            quote! {
                required_scratch = required_scratch.max(
                    <B as ember_core::KernelBackend>::conv2d_scratch_size(
                        #input_shape,
                        #weights_shape,
                        #output_shape,
                    )
                );
            }
        }
        BuiltinOperator::DEPTHWISE_CONV_2D => {
            let inputs = operator.inputs().unwrap();
            let input_shape = shape4_tokens(shape4(
                &tensor_info(tensors.get(inputs.get(0) as usize)),
                "DepthwiseConv2D input",
            ));
            let weights_shape = shape4_tokens(shape4(
                &tensor_info(tensors.get(inputs.get(1) as usize)),
                "DepthwiseConv2D weights",
            ));
            let output_shape = shape4_tokens(shape4(
                &tensor_info(tensors.get(operator.outputs().unwrap().get(0) as usize)),
                "DepthwiseConv2D output",
            ));
            quote! {
                required_scratch = required_scratch.max(
                    <B as ember_core::KernelBackend>::depthwise_conv2d_scratch_size(
                        #input_shape,
                        #weights_shape,
                        #output_shape,
                    )
                );
            }
        }
        BuiltinOperator::SOFTMAX => {
            let inputs = operator.inputs().unwrap();
            let input_info = tensor_info(tensors.get(inputs.get(0) as usize));
            let input_shape = shape2(&input_info, "Softmax input");
            let num_classes = input_shape[1];
            quote! {
                required_scratch = required_scratch.max(
                    <B as ember_core::KernelBackend>::softmax_scratch_size(#num_classes)
                );
            }
        }
        _ => quote!(),
    }
}

fn activation_tokens(activation: ActivationFunctionType) -> TokenStream2 {
    match activation {
        ActivationFunctionType::NONE => quote!(ember_core::FusedActivation::None),
        ActivationFunctionType::RELU => quote!(ember_core::FusedActivation::Relu),
        ActivationFunctionType::RELU6 => quote!(ember_core::FusedActivation::Relu6),
        ActivationFunctionType::RELU_N1_TO_1 => quote!(ember_core::FusedActivation::ReluN1To1),
        ActivationFunctionType::TANH => quote!(ember_core::FusedActivation::Tanh),
        ActivationFunctionType::SIGN_BIT => quote!(ember_core::FusedActivation::SignBit),
        ActivationFunctionType::SIGMOID => quote!(ember_core::FusedActivation::Sigmoid),
        unsupported => abort_call_site!("unsupported fused activation: {:?}", unsupported),
    }
}

fn padding_tokens(padding: Padding) -> TokenStream2 {
    match padding {
        Padding::SAME => quote!(ember_core::Padding::Same),
        Padding::VALID => quote!(ember_core::Padding::Valid),
        unsupported => abort_call_site!("unsupported padding: {:?}", unsupported),
    }
}

fn tensor_ref(tensor_id: usize, input_tensor_id: usize) -> TokenStream2 {
    if tensor_id == input_tensor_id {
        quote!(input)
    } else {
        let ident = format_ident!("tensor_{}", tensor_id);
        quote!(&#ident)
    }
}

fn tensor_mut_ref(tensor_id: usize, output_tensor_id: usize) -> TokenStream2 {
    if tensor_id == output_tensor_id {
        quote!(output)
    } else {
        let ident = format_ident!("tensor_{}", tensor_id);
        quote!(&mut #ident)
    }
}

fn maybe_alloc_output(
    output_id: usize,
    output_tensor_id: usize,
    output_len: usize,
) -> TokenStream2 {
    if output_id == output_tensor_id {
        quote!()
    } else {
        let ident = format_ident!("tensor_{}", output_id);
        quote!(let mut #ident = [0i8; #output_len];)
    }
}

fn const_i8_array(name: &proc_macro2::Ident, values: &[i8]) -> TokenStream2 {
    let len = values.len();
    quote!(const #name: [i8; #len] = [#(#values),*];)
}

fn const_i32_array(name: &proc_macro2::Ident, values: &[i32]) -> TokenStream2 {
    let len = values.len();
    quote!(const #name: [i32; #len] = [#(#values),*];)
}

fn bias_tokens(
    index: usize,
    maybe_tensor_id: Option<usize>,
    tensors: &flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<Tensor<'_>>>,
    buffers: &flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<Buffer<'_>>>,
) -> (TokenStream2, TokenStream2) {
    if let Some(tensor_id) = maybe_tensor_id {
        let ident = format_ident!("BIAS_{}", index);
        let values = tensor_i32_data(tensors.get(tensor_id), buffers);
        let decl = const_i32_array(&ident, &values);
        (decl, quote!(Some(&#ident)))
    } else {
        (quote!(), quote!(None))
    }
}

fn emit_conv2d(
    index: usize,
    operator: tflite_flatbuffers::tflite::Operator<'_>,
    tensors: &flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<Tensor<'_>>>,
    buffers: &flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<Buffer<'_>>>,
    input_tensor_id: usize,
    output_tensor_id: usize,
) -> TokenStream2 {
    let inputs = operator.inputs().unwrap();
    let input_id = inputs.get(0) as usize;
    let weights_id = inputs.get(1) as usize;
    let bias_id = (inputs.len() > 2 && inputs.get(2) >= 0).then(|| inputs.get(2) as usize);
    let output_id = operator.outputs().unwrap().get(0) as usize;

    let input_info = tensor_info(tensors.get(input_id));
    let weights_info = tensor_info(tensors.get(weights_id));
    let output_info = tensor_info(tensors.get(output_id));
    require_i8(&input_info, "Conv2D input");
    require_i8(&weights_info, "Conv2D weights");
    require_i8(&output_info, "Conv2D output");

    let weights_ident = format_ident!("WEIGHTS_{}", index);
    let weights_values = tensor_i8_data(tensors.get(weights_id), buffers);
    let weights_decl = const_i8_array(&weights_ident, &weights_values);
    let (bias_decl, bias_expr) = bias_tokens(index, bias_id, tensors, buffers);

    let output_len = tensor_len(&output_info);
    let alloc_output = maybe_alloc_output(output_id, output_tensor_id, output_len);
    let input_expr = tensor_ref(input_id, input_tensor_id);
    let output_expr = tensor_mut_ref(output_id, output_tensor_id);
    let input_shape = shape4_tokens(shape4(&input_info, "Conv2D input"));
    let weights_shape = shape4_tokens(shape4(&weights_info, "Conv2D weights"));
    let output_shape = shape4_tokens(shape4(&output_info, "Conv2D output"));
    let input_quant = quant_tokens(&input_info);
    let weights_quant = quant_tokens(&weights_info);
    let output_quant = quant_tokens(&output_info);
    let options = operator.builtin_options_as_conv_2_doptions().unwrap();
    let padding = padding_tokens(options.padding());
    let activation = activation_tokens(options.fused_activation_function());
    let stride_w = options.stride_w();
    let stride_h = options.stride_h();
    let dilation_w_factor = options.dilation_w_factor();
    let dilation_h_factor = options.dilation_h_factor();

    quote! {
        #weights_decl
        #bias_decl
        #alloc_output
        backend.conv2d(ember_core::Conv2dParams {
            input: #input_expr,
            input_shape: #input_shape,
            input_quant: #input_quant,
            weights: &#weights_ident,
            weights_shape: #weights_shape,
            weights_quant: #weights_quant,
            bias: #bias_expr,
            output: #output_expr,
            output_shape: #output_shape,
            output_quant: #output_quant,
            stride_w: #stride_w,
            stride_h: #stride_h,
            dilation_w_factor: #dilation_w_factor,
            dilation_h_factor: #dilation_h_factor,
            padding: #padding,
            activation: #activation,
            scratch: &mut *scratch,
        })?;
    }
}

fn emit_depthwise_conv2d(
    index: usize,
    operator: tflite_flatbuffers::tflite::Operator<'_>,
    tensors: &flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<Tensor<'_>>>,
    buffers: &flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<Buffer<'_>>>,
    input_tensor_id: usize,
    output_tensor_id: usize,
) -> TokenStream2 {
    let inputs = operator.inputs().unwrap();
    let input_id = inputs.get(0) as usize;
    let weights_id = inputs.get(1) as usize;
    let bias_id = (inputs.len() > 2 && inputs.get(2) >= 0).then(|| inputs.get(2) as usize);
    let output_id = operator.outputs().unwrap().get(0) as usize;

    let input_info = tensor_info(tensors.get(input_id));
    let weights_info = tensor_info(tensors.get(weights_id));
    let output_info = tensor_info(tensors.get(output_id));
    require_i8(&input_info, "DepthwiseConv2D input");
    require_i8(&weights_info, "DepthwiseConv2D weights");
    require_i8(&output_info, "DepthwiseConv2D output");

    let weights_ident = format_ident!("WEIGHTS_{}", index);
    let weights_values = tensor_i8_data(tensors.get(weights_id), buffers);
    let weights_decl = const_i8_array(&weights_ident, &weights_values);
    let (bias_decl, bias_expr) = bias_tokens(index, bias_id, tensors, buffers);

    let output_len = tensor_len(&output_info);
    let alloc_output = maybe_alloc_output(output_id, output_tensor_id, output_len);
    let input_expr = tensor_ref(input_id, input_tensor_id);
    let output_expr = tensor_mut_ref(output_id, output_tensor_id);
    let input_shape = shape4_tokens(shape4(&input_info, "DepthwiseConv2D input"));
    let weights_shape = shape4_tokens(shape4(&weights_info, "DepthwiseConv2D weights"));
    let output_shape = shape4_tokens(shape4(&output_info, "DepthwiseConv2D output"));
    let input_quant = quant_tokens(&input_info);
    let weights_quant = quant_tokens(&weights_info);
    let output_quant = quant_tokens(&output_info);
    let options = operator
        .builtin_options_as_depthwise_conv_2_doptions()
        .unwrap();
    let padding = padding_tokens(options.padding());
    let activation = activation_tokens(options.fused_activation_function());
    let stride_w = options.stride_w();
    let stride_h = options.stride_h();
    let dilation_w_factor = options.dilation_w_factor();
    let dilation_h_factor = options.dilation_h_factor();
    let depth_multiplier = options.depth_multiplier();

    quote! {
        #weights_decl
        #bias_decl
        #alloc_output
        backend.depthwise_conv2d(ember_core::DepthwiseConv2dParams {
            input: #input_expr,
            input_shape: #input_shape,
            input_quant: #input_quant,
            weights: &#weights_ident,
            weights_shape: #weights_shape,
            weights_quant: #weights_quant,
            bias: #bias_expr,
            output: #output_expr,
            output_shape: #output_shape,
            output_quant: #output_quant,
            stride_w: #stride_w,
            stride_h: #stride_h,
            dilation_w_factor: #dilation_w_factor,
            dilation_h_factor: #dilation_h_factor,
            depth_multiplier: #depth_multiplier,
            padding: #padding,
            activation: #activation,
            scratch: &mut *scratch,
        })?;
    }
}

fn emit_fully_connected(
    index: usize,
    operator: tflite_flatbuffers::tflite::Operator<'_>,
    tensors: &flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<Tensor<'_>>>,
    buffers: &flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<Buffer<'_>>>,
    input_tensor_id: usize,
    output_tensor_id: usize,
) -> TokenStream2 {
    let inputs = operator.inputs().unwrap();
    let input_id = inputs.get(0) as usize;
    let weights_id = inputs.get(1) as usize;
    let bias_id = (inputs.len() > 2 && inputs.get(2) >= 0).then(|| inputs.get(2) as usize);
    let output_id = operator.outputs().unwrap().get(0) as usize;

    let input_info = tensor_info(tensors.get(input_id));
    let weights_info = tensor_info(tensors.get(weights_id));
    let output_info = tensor_info(tensors.get(output_id));
    require_i8(&input_info, "FullyConnected input");
    require_i8(&weights_info, "FullyConnected weights");
    require_i8(&output_info, "FullyConnected output");

    let weights_ident = format_ident!("WEIGHTS_{}", index);
    let weights_values = tensor_i8_data(tensors.get(weights_id), buffers);
    let weights_decl = const_i8_array(&weights_ident, &weights_values);
    let (bias_decl, bias_expr) = bias_tokens(index, bias_id, tensors, buffers);

    let output_len = tensor_len(&output_info);
    let alloc_output = maybe_alloc_output(output_id, output_tensor_id, output_len);
    let input_expr = tensor_ref(input_id, input_tensor_id);
    let output_expr = tensor_mut_ref(output_id, output_tensor_id);
    let weights_shape = shape2_tokens(shape2(&weights_info, "FullyConnected weights"));
    let input_quant = quant_tokens(&input_info);
    let weights_quant = quant_tokens(&weights_info);
    let output_quant = quant_tokens(&output_info);
    let options = operator
        .builtin_options_as_fully_connected_options()
        .unwrap();
    let activation = activation_tokens(options.fused_activation_function());

    quote! {
        #weights_decl
        #bias_decl
        #alloc_output
        backend.fully_connected(ember_core::FullyConnectedParams {
            input: #input_expr,
            input_quant: #input_quant,
            weights: &#weights_ident,
            weights_shape: #weights_shape,
            weights_quant: #weights_quant,
            bias: #bias_expr,
            output: #output_expr,
            output_depth: #output_len,
            output_quant: #output_quant,
            activation: #activation,
        })?;
    }
}

fn emit_pool(
    _index: usize,
    operator: tflite_flatbuffers::tflite::Operator<'_>,
    tensors: &flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<Tensor<'_>>>,
    input_tensor_id: usize,
    output_tensor_id: usize,
    average: bool,
) -> TokenStream2 {
    let inputs = operator.inputs().unwrap();
    let input_id = inputs.get(0) as usize;
    let output_id = operator.outputs().unwrap().get(0) as usize;
    let input_info = tensor_info(tensors.get(input_id));
    let output_info = tensor_info(tensors.get(output_id));
    require_i8(&input_info, "Pool input");
    require_i8(&output_info, "Pool output");

    let output_len = tensor_len(&output_info);
    let alloc_output = maybe_alloc_output(output_id, output_tensor_id, output_len);
    let input_expr = tensor_ref(input_id, input_tensor_id);
    let output_expr = tensor_mut_ref(output_id, output_tensor_id);
    let input_shape = shape4_tokens(shape4(&input_info, "Pool input"));
    let output_shape = shape4_tokens(shape4(&output_info, "Pool output"));
    let input_quant = quant_tokens(&input_info);
    let output_quant = quant_tokens(&output_info);
    let options = operator.builtin_options_as_pool_2_doptions().unwrap();
    let padding = padding_tokens(options.padding());
    let activation = activation_tokens(options.fused_activation_function());
    let stride_w = options.stride_w();
    let stride_h = options.stride_h();
    let filter_w = options.filter_width();
    let filter_h = options.filter_height();
    let method = if average {
        quote!(avg_pool)
    } else {
        quote!(max_pool)
    };

    quote! {
        #alloc_output
        backend.#method(ember_core::PoolParams {
            input: #input_expr,
            input_shape: #input_shape,
            input_quant: #input_quant,
            output: #output_expr,
            output_shape: #output_shape,
            output_quant: #output_quant,
            stride_w: #stride_w,
            stride_h: #stride_h,
            filter_w: #filter_w,
            filter_h: #filter_h,
            padding: #padding,
            activation: #activation,
        })?;
    }
}

fn emit_softmax(
    _index: usize,
    operator: tflite_flatbuffers::tflite::Operator<'_>,
    tensors: &flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<Tensor<'_>>>,
    input_tensor_id: usize,
    output_tensor_id: usize,
) -> TokenStream2 {
    let inputs = operator.inputs().unwrap();
    let input_id = inputs.get(0) as usize;
    let output_id = operator.outputs().unwrap().get(0) as usize;
    let input_info = tensor_info(tensors.get(input_id));
    let output_info = tensor_info(tensors.get(output_id));
    require_i8(&input_info, "Softmax input");
    require_i8(&output_info, "Softmax output");

    let output_len = tensor_len(&output_info);
    let alloc_output = maybe_alloc_output(output_id, output_tensor_id, output_len);
    let input_expr = tensor_ref(input_id, input_tensor_id);
    let output_expr = tensor_mut_ref(output_id, output_tensor_id);
    let input_shape = shape2_tokens(shape2(&input_info, "Softmax input"));
    let input_quant = quant_tokens(&input_info);
    let output_quant = quant_tokens(&output_info);
    let beta = operator
        .builtin_options_as_softmax_options()
        .unwrap()
        .beta();

    quote! {
        #alloc_output
        backend.softmax(ember_core::SoftmaxParams {
            input: #input_expr,
            input_shape: #input_shape,
            input_quant: #input_quant,
            output: #output_expr,
            output_quant: #output_quant,
            beta: #beta,
            scratch: &mut *scratch,
        })?;
    }
}

fn emit_add(
    operator: tflite_flatbuffers::tflite::Operator<'_>,
    tensors: &flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<Tensor<'_>>>,
    input_tensor_id: usize,
    output_tensor_id: usize,
) -> TokenStream2 {
    let inputs = operator.inputs().unwrap();
    let input1_id = inputs.get(0) as usize;
    let input2_id = inputs.get(1) as usize;
    let output_id = operator.outputs().unwrap().get(0) as usize;

    let input1_info = tensor_info(tensors.get(input1_id));
    let input2_info = tensor_info(tensors.get(input2_id));
    let output_info = tensor_info(tensors.get(output_id));
    require_i8(&input1_info, "ADD input1");
    require_i8(&input2_info, "ADD input2");
    require_i8(&output_info, "ADD output");

    let output_len = tensor_len(&output_info);
    let alloc_output = maybe_alloc_output(output_id, output_tensor_id, output_len);
    let input1_expr = tensor_ref(input1_id, input_tensor_id);
    let input2_expr = tensor_ref(input2_id, input_tensor_id);
    let output_expr = tensor_mut_ref(output_id, output_tensor_id);
    let input1_quant = quant_tokens(&input1_info);
    let input2_quant = quant_tokens(&input2_info);
    let output_quant = quant_tokens(&output_info);
    let activation = activation_tokens(
        operator
            .builtin_options_as_add_options()
            .unwrap()
            .fused_activation_function(),
    );

    quote! {
        #alloc_output
        backend.add(ember_core::ElementwiseAddParams {
            input1: #input1_expr,
            input1_quant: #input1_quant,
            input2: #input2_expr,
            input2_quant: #input2_quant,
            output: #output_expr,
            output_quant: #output_quant,
            activation: #activation,
        })?;
    }
}

fn emit_reshape(
    operator: tflite_flatbuffers::tflite::Operator<'_>,
    tensors: &flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<Tensor<'_>>>,
    input_tensor_id: usize,
    output_tensor_id: usize,
) -> TokenStream2 {
    let inputs = operator.inputs().unwrap();
    let input_id = inputs.get(0) as usize;
    let output_id = operator.outputs().unwrap().get(0) as usize;
    let input_expr = tensor_ref(input_id, input_tensor_id);
    let output_info = tensor_info(tensors.get(output_id));
    let output_len = tensor_len(&output_info);

    if output_id == output_tensor_id {
        quote!(output.copy_from_slice(#input_expr);)
    } else {
        let ident = format_ident!("tensor_{}", output_id);
        quote! {
            let mut #ident = [0i8; #output_len];
            #ident.copy_from_slice(#input_expr);
        }
    }
}
