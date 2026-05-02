#![no_std]
#![deny(missing_docs)]

//! # ember-core
//!
//! Core trait definitions for the ember-rs embedded TinyML inference engine.
//!
//! ember-rs is a `no_std` INT8 inference engine designed to be the
//! "Burn of embedded inference" - providing a pluggable [`KernelBackend`] trait
//! so different hardware backends can be swapped without changing model code.
//!
//! ## Design
//!
//! - Param struct field names mirror TFLite Micro's C structs
//!   (`TfLiteConvParams`, `TfLiteDepthwiseConvParams`, etc.)
//! - The `invoke` phase is covered by this trait; the `prepare` phase
//!   (scratch size calculation, shape inference) is handled at compile time
//!   by `ember-macros`
//! - `scratch_size_*` functions have default implementations returning `0`,
//!   so pure-Rust reference backends don't need to implement them
//!
//! ## Backend implementations
//!
//! - `ember-ref`: pure Rust reference implementation (for testing / non-ESP platforms)
//! - `ember-esp`: official ESP32-S3 backend using Espressif's esp-nn SIMD kernels
//!   (maintained in a separate repository)

// ----------------------------------------------------------------------------
// Enums - mirror TFLite Micro's C enums
// ----------------------------------------------------------------------------

/// Padding strategy, mirrors `TfLitePadding`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Padding {
    /// Output size equals `ceil(input_size / stride)`.
    Same,
    /// No padding; output size equals `floor((input_size - filter_size) / stride) + 1`.
    Valid,
}

/// Fused activation function applied after an operator, mirrors `TfLiteFusedActivation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedActivation {
    /// No activation.
    None,
    /// ReLU: `max(0, x)`.
    Relu,
    /// ReLU6: `min(max(0, x), 6)`.
    Relu6,
    /// ReLU with range `[-1, 1]`.
    ReluN1To1,
    /// Tanh activation.
    Tanh,
    /// Sign bit activation.
    SignBit,
    /// Sigmoid activation.
    Sigmoid,
}

// ----------------------------------------------------------------------------
// Quantization params - mirror TfLiteQuantizationParams
// ----------------------------------------------------------------------------

/// Per-tensor quantization parameters, mirrors `TfLiteQuantizationParams`.
#[derive(Clone, Copy, Debug)]
pub struct QuantParam {
    /// The scale factor: `real_value = scale * (quantized_value - zero_point)`.
    pub scale: f32,
    /// The zero point for asymmetric quantization.
    pub zero_point: i32,
}

// ----------------------------------------------------------------------------
// Error / Status - mirror TfLiteStatus
// ----------------------------------------------------------------------------

/// Errors that a [`KernelBackend`] implementation may return.
/// Mirrors the error states of `TfLiteStatus`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelError {
    /// Input or output tensor shapes are invalid for this operation.
    InvalidShape,
    /// The requested activation function is not supported by this backend.
    UnsupportedActivation,
    /// A buffer passed to the backend does not meet alignment requirements.
    ///
    /// ESP-NN kernels require 16-byte aligned buffers. If the `assume-aligned`
    /// feature is disabled (default), misaligned buffers are automatically
    /// copied to an aligned scratch region. This error is only returned when
    /// `assume-aligned` is enabled and the caller violates the contract.
    AlignmentError,
    /// An internal error occurred in the backend.
    InternalError,
}

/// Result type for all [`KernelBackend`] operations, mirrors `TfLiteStatus`.
pub type Status = Result<(), KernelError>;

// ----------------------------------------------------------------------------
// Operator parameter structs
// Field names mirror TFLite Micro's C structs for easy cross-reference.
// ----------------------------------------------------------------------------

/// Parameters for a 2D convolution operation.
///
/// Mirrors `TfLiteConvParams` from TFLite Micro.
/// Tensor layout: NHWC (batch, height, width, channels).
pub struct Conv2dParams<'a> {
    /// Input tensor data, quantized as `int8`.
    pub input: &'a [i8],
    /// Input tensor shape `[N, H, W, C_in]`.
    pub input_shape: [usize; 4],
    /// Input quantization parameters.
    pub input_quant: QuantParam,
    /// Weight tensor data, quantized as `int8`. Layout: `[C_out, KH, KW, C_in]`.
    pub weights: &'a [i8],
    /// Weight tensor shape `[C_out, KH, KW, C_in]`.
    pub weights_shape: [usize; 4],
    /// Weight quantization parameters (per-tensor).
    pub weights_quant: QuantParam,
    /// Optional bias tensor, stored as `int32`.
    ///
    /// Length must equal `C_out` when `Some`.
    pub bias: Option<&'a [i32]>,
    /// Output tensor buffer, written by the backend.
    pub output: &'a mut [i8],
    /// Output tensor shape `[N, H_out, W_out, C_out]`.
    pub output_shape: [usize; 4],
    /// Output quantization parameters.
    pub output_quant: QuantParam,
    /// Horizontal stride.
    pub stride_w: i32,
    /// Vertical stride.
    pub stride_h: i32,
    /// Horizontal dilation factor (1 = no dilation).
    pub dilation_w_factor: i32,
    /// Vertical dilation factor (1 = no dilation).
    pub dilation_h_factor: i32,
    /// Padding mode.
    pub padding: Padding,
    /// Fused activation function applied to the output.
    pub activation: FusedActivation,
    /// Scratch buffer for intermediate computation.
    ///
    /// Required size is reported by [`KernelBackend::conv2d_scratch_size`].
    /// Pass an empty slice `&mut []` if the backend does not require scratch memory.
    pub scratch: &'a mut [u8],
}

/// Parameters for a depthwise 2D convolution operation.
///
/// Mirrors `TfLiteDepthwiseConvParams` from TFLite Micro.
pub struct DepthwiseConv2dParams<'a> {
    /// Input tensor data, quantized as `int8`.
    pub input: &'a [i8],
    /// Input tensor shape `[N, H, W, C_in]`.
    pub input_shape: [usize; 4],
    /// Input quantization parameters.
    pub input_quant: QuantParam,
    /// Weight tensor data, quantized as `int8`.
    pub weights: &'a [i8],
    /// Weight tensor shape.
    pub weights_shape: [usize; 4],
    /// Weight quantization parameters (per-tensor).
    pub weights_quant: QuantParam,
    /// Optional bias tensor, stored as `int32`.
    pub bias: Option<&'a [i32]>,
    /// Output tensor buffer, written by the backend.
    pub output: &'a mut [i8],
    /// Output tensor shape `[N, H_out, W_out, C_out]`.
    pub output_shape: [usize; 4],
    /// Output quantization parameters.
    pub output_quant: QuantParam,
    /// Horizontal stride.
    pub stride_w: i32,
    /// Vertical stride.
    pub stride_h: i32,
    /// Horizontal dilation factor (1 = no dilation).
    pub dilation_w_factor: i32,
    /// Vertical dilation factor (1 = no dilation).
    pub dilation_h_factor: i32,
    /// Depth multiplier - the number of output channels per input channel.
    ///
    /// Specific to depthwise convolution; mirrors
    /// `TfLiteDepthwiseConvParams::depth_multiplier`.
    pub depth_multiplier: i32,
    /// Padding mode.
    pub padding: Padding,
    /// Fused activation function applied to the output.
    pub activation: FusedActivation,
    /// Scratch buffer for intermediate computation.
    pub scratch: &'a mut [u8],
}

/// Parameters for a fully-connected (dense) layer.
///
/// Mirrors `TfLiteFullyConnectedParams` from TFLite Micro.
pub struct FullyConnectedParams<'a> {
    /// Input tensor data, quantized as `int8`.
    pub input: &'a [i8],
    /// Input quantization parameters.
    pub input_quant: QuantParam,
    /// Weight tensor data. Layout: `[output_depth, input_depth]`.
    pub weights: &'a [i8],
    /// Weight tensor shape `[output_depth, input_depth]`.
    pub weights_shape: [usize; 2],
    /// Weight quantization parameters (per-tensor).
    pub weights_quant: QuantParam,
    /// Optional bias tensor, stored as `int32`.
    pub bias: Option<&'a [i32]>,
    /// Output tensor buffer, written by the backend.
    pub output: &'a mut [i8],
    /// Number of output neurons.
    pub output_depth: usize,
    /// Output quantization parameters.
    pub output_quant: QuantParam,
    /// Fused activation function applied to the output.
    pub activation: FusedActivation,
}

/// Parameters for a pooling operation (average or max).
///
/// Mirrors `TfLitePoolParams` from TFLite Micro.
pub struct PoolParams<'a> {
    /// Input tensor data, quantized as `int8`.
    pub input: &'a [i8],
    /// Input tensor shape `[N, H, W, C]`.
    pub input_shape: [usize; 4],
    /// Input quantization parameters.
    pub input_quant: QuantParam,
    /// Output tensor buffer, written by the backend.
    pub output: &'a mut [i8],
    /// Output tensor shape `[N, H_out, W_out, C]`.
    pub output_shape: [usize; 4],
    /// Output quantization parameters.
    pub output_quant: QuantParam,
    /// Horizontal stride.
    pub stride_w: i32,
    /// Vertical stride.
    pub stride_h: i32,
    /// Pooling filter width.
    pub filter_w: i32,
    /// Pooling filter height.
    pub filter_h: i32,
    /// Padding mode.
    pub padding: Padding,
    /// Fused activation function applied to the output.
    pub activation: FusedActivation,
}

/// Parameters for the softmax operation.
///
/// Mirrors `TfLiteSoftmaxParams` from TFLite Micro.
pub struct SoftmaxParams<'a> {
    /// Input tensor data, quantized as `int8`.
    pub input: &'a [i8],
    /// Input shape `[batch, num_classes]`.
    pub input_shape: [usize; 2],
    /// Input quantization parameters.
    pub input_quant: QuantParam,
    /// Output tensor buffer, written by the backend.
    pub output: &'a mut [i8],
    /// Output quantization parameters.
    pub output_quant: QuantParam,
    /// Softmax beta parameter (typically `1.0`).
    ///
    /// Mirrors `TfLiteSoftmaxParams::beta`.
    pub beta: f32,
    /// Scratch buffer for intermediate computation.
    pub scratch: &'a mut [u8],
}

/// Parameters for element-wise addition.
///
/// Mirrors `TfLiteAddParams` from TFLite Micro.
pub struct ElementwiseAddParams<'a> {
    /// First input tensor data, quantized as `int8`.
    pub input1: &'a [i8],
    /// First input quantization parameters.
    pub input1_quant: QuantParam,
    /// Second input tensor data, quantized as `int8`.
    pub input2: &'a [i8],
    /// Second input quantization parameters.
    pub input2_quant: QuantParam,
    /// Output tensor buffer, written by the backend.
    pub output: &'a mut [i8],
    /// Output quantization parameters.
    pub output_quant: QuantParam,
    /// Fused activation function applied to the output.
    pub activation: FusedActivation,
}

// ----------------------------------------------------------------------------
// KernelBackend - the central trait
// ----------------------------------------------------------------------------

/// The core abstraction for ember-rs: a hardware-specific INT8 inference backend.
///
/// # Design
///
/// This trait covers the **invoke phase** only. The **prepare phase**
/// (scratch buffer sizing, shape inference) is performed at compile time by
/// `ember-macros` via the `conv2d_scratch_size` / `softmax_scratch_size`
/// associated functions, which have default implementations returning `0`.
///
/// Implementations map directly onto TFLite Micro kernel `invoke` functions,
/// which means porting an existing TFLite Micro optimized kernel (e.g., CMSIS-NN,
/// esp-nn) to ember-rs requires minimal glue code.
///
/// # Implementing a backend
///
/// ```rust,ignore
/// use ember_core::{KernelBackend, Conv2dParams, Status};
///
/// pub struct MyBackend;
///
/// impl KernelBackend for MyBackend {
///     fn conv2d(&mut self, params: Conv2dParams<'_>) -> Status {
///         // call your hardware-accelerated kernel here
///         todo!()
///     }
///     // ... implement remaining required methods
/// }
/// ```
///
/// # Scratch buffers
///
/// Backends that require scratch memory (e.g., esp-nn, CMSIS-NN) must override
/// the `*_scratch_size` associated functions. The `ember-macros` proc macro calls
/// these at compile time to allocate correctly-sized scratch arrays in the
/// generated inference function.
pub trait KernelBackend {
    /// Execute a 2D convolution.
    ///
    /// Corresponds to the `invoke` function of the `CONV_2D` kernel in TFLite Micro.
    fn conv2d(&mut self, params: Conv2dParams<'_>) -> Status;

    /// Execute a depthwise 2D convolution.
    ///
    /// Corresponds to the `invoke` function of the `DEPTHWISE_CONV_2D` kernel
    /// in TFLite Micro.
    fn depthwise_conv2d(&mut self, params: DepthwiseConv2dParams<'_>) -> Status;

    /// Execute a fully-connected layer.
    ///
    /// Corresponds to the `invoke` function of the `FULLY_CONNECTED` kernel
    /// in TFLite Micro.
    fn fully_connected(&mut self, params: FullyConnectedParams<'_>) -> Status;

    /// Execute average pooling.
    ///
    /// Corresponds to the `invoke` function of the `AVERAGE_POOL_2D` kernel
    /// in TFLite Micro.
    fn avg_pool(&mut self, params: PoolParams<'_>) -> Status;

    /// Execute max pooling.
    ///
    /// Corresponds to the `invoke` function of the `MAX_POOL_2D` kernel
    /// in TFLite Micro.
    fn max_pool(&mut self, params: PoolParams<'_>) -> Status;

    /// Execute softmax.
    ///
    /// Corresponds to the `invoke` function of the `SOFTMAX` kernel
    /// in TFLite Micro.
    fn softmax(&mut self, params: SoftmaxParams<'_>) -> Status;

    /// Execute element-wise addition.
    ///
    /// Corresponds to the `invoke` function of the `ADD` kernel in TFLite Micro.
    fn add(&mut self, params: ElementwiseAddParams<'_>) -> Status;

    /// Returns the scratch buffer size in bytes required by [`Self::conv2d`].
    ///
    /// Called by `ember-macros` at **compile time** to allocate scratch arrays
    /// in the generated inference function. Corresponds to
    /// `esp_nn_get_conv_scratch_size` / CMSIS-NN equivalents.
    ///
    /// The default implementation returns `0` (no scratch required), which is
    /// correct for pure-Rust reference backends.
    fn conv2d_scratch_size(
        input_shape: [usize; 4],
        weights_shape: [usize; 4],
        output_shape: [usize; 4],
    ) -> usize
    where
        Self: Sized,
    {
        let _ = (input_shape, weights_shape, output_shape);
        0
    }

    /// Returns the scratch buffer size in bytes required by [`Self::depthwise_conv2d`].
    fn depthwise_conv2d_scratch_size(
        input_shape: [usize; 4],
        weights_shape: [usize; 4],
        output_shape: [usize; 4],
    ) -> usize
    where
        Self: Sized,
    {
        let _ = (input_shape, weights_shape, output_shape);
        0
    }

    /// Returns the scratch buffer size in bytes required by [`Self::softmax`].
    fn softmax_scratch_size(num_classes: usize) -> usize
    where
        Self: Sized,
    {
        let _ = num_classes;
        0
    }
}
