#![no_std]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use ember_infer_core::{
    Conv2dParams, DepthwiseConv2dParams, ElementwiseAddParams, FullyConnectedParams,
    FusedActivation, KernelBackend, KernelError, MulParams, Padding, PerChannelQuantParam,
    PoolParams, QuantParam, SoftmaxParams, Status,
};

/// Pure Rust reference implementation of [`KernelBackend`].
///
/// Used for CI testing and platforms where no hardware-accelerated backend is
/// available.
pub struct RefBackend;

impl KernelBackend for RefBackend {
    fn conv2d(&mut self, params: Conv2dParams<'_>) -> Status {
        validate_len(params.input, product(&params.input_shape))?;
        validate_len(params.weights, product(&params.weights_shape))?;
        validate_len(params.output, product(&params.output_shape))?;

        let [batches, input_h, input_w, input_c] = params.input_shape;
        let [output_c, filter_h, filter_w, filter_input_c] = params.weights_shape;
        let [output_batches, output_h, output_w, output_shape_c] = params.output_shape;

        if batches != output_batches || input_c != filter_input_c || output_c != output_shape_c {
            return Err(KernelError::InvalidShape);
        }
        validate_bias(params.bias, output_c)?;

        let stride_h = positive_i32_to_usize(params.stride_h)?;
        let stride_w = positive_i32_to_usize(params.stride_w)?;
        let dilation_h = positive_i32_to_usize(params.dilation_h_factor)?;
        let dilation_w = positive_i32_to_usize(params.dilation_w_factor)?;
        let effective_filter_h = effective_filter_size(filter_h, dilation_h);
        let effective_filter_w = effective_filter_size(filter_w, dilation_w);
        let pad_h = compute_padding(input_h, effective_filter_h, stride_h, params.padding);
        let pad_w = compute_padding(input_w, effective_filter_w, stride_w, params.padding);
        for batch in 0..batches {
            for out_y in 0..output_h {
                for out_x in 0..output_w {
                    for out_channel in 0..output_c {
                        let (multiplier, shift) = output_channel_multiplier_shift(
                            params.input_quant,
                            params.weights_quant,
                            params.weights_per_channel_quant,
                            params.output_quant,
                            out_channel,
                        );
                        let mut acc = params
                            .bias
                            .map(|bias| bias[out_channel])
                            .unwrap_or_default();

                        for filter_y in 0..filter_h {
                            let in_y = out_y * stride_h + filter_y * dilation_h;
                            if in_y < pad_h || in_y >= input_h + pad_h {
                                continue;
                            }
                            let in_y = in_y - pad_h;

                            for filter_x in 0..filter_w {
                                let in_x = out_x * stride_w + filter_x * dilation_w;
                                if in_x < pad_w || in_x >= input_w + pad_w {
                                    continue;
                                }
                                let in_x = in_x - pad_w;

                                for in_channel in 0..input_c {
                                    let input = params.input[nhwc_index(
                                        batch, in_y, in_x, in_channel, input_h, input_w, input_c,
                                    )] as i32
                                        - params.input_quant.zero_point;
                                    let weight = params.weights[conv_weight_index(
                                        out_channel,
                                        filter_y,
                                        filter_x,
                                        in_channel,
                                        filter_h,
                                        filter_w,
                                        input_c,
                                    )] as i32
                                        - params.weights_quant.zero_point;
                                    acc = acc.saturating_add(input.saturating_mul(weight));
                                }
                            }
                        }

                        let scaled = requantize(acc, multiplier, shift, params.output_quant);
                        params.output[nhwc_index(
                            batch,
                            out_y,
                            out_x,
                            out_channel,
                            output_h,
                            output_w,
                            output_c,
                        )] = apply_activation(scaled, params.activation, params.output_quant);
                    }
                }
            }
        }

        Ok(())
    }

    fn depthwise_conv2d(&mut self, params: DepthwiseConv2dParams<'_>) -> Status {
        validate_len(params.input, product(&params.input_shape))?;
        validate_len(params.weights, product(&params.weights_shape))?;
        validate_len(params.output, product(&params.output_shape))?;

        let [batches, input_h, input_w, input_c] = params.input_shape;
        let depth_multiplier = positive_i32_to_usize(params.depth_multiplier)?;
        let depthwise_dims =
            depthwise_filter_dims(params.weights_shape, input_c, depth_multiplier)?;
        let [output_batches, output_h, output_w, output_c] = params.output_shape;

        if batches != output_batches
            || input_c != depthwise_dims.input_channels
            || depth_multiplier != depthwise_dims.depth_multiplier
            || output_c != input_c * depth_multiplier
        {
            return Err(KernelError::InvalidShape);
        }
        validate_bias(params.bias, output_c)?;

        let stride_h = positive_i32_to_usize(params.stride_h)?;
        let stride_w = positive_i32_to_usize(params.stride_w)?;
        let dilation_h = positive_i32_to_usize(params.dilation_h_factor)?;
        let dilation_w = positive_i32_to_usize(params.dilation_w_factor)?;
        let effective_filter_h = effective_filter_size(depthwise_dims.filter_h, dilation_h);
        let effective_filter_w = effective_filter_size(depthwise_dims.filter_w, dilation_w);
        let pad_h = compute_padding(input_h, effective_filter_h, stride_h, params.padding);
        let pad_w = compute_padding(input_w, effective_filter_w, stride_w, params.padding);
        for batch in 0..batches {
            for out_y in 0..output_h {
                for out_x in 0..output_w {
                    for in_channel in 0..input_c {
                        for channel_multiplier in 0..depth_multiplier {
                            let out_channel = in_channel * depth_multiplier + channel_multiplier;
                            let (multiplier, shift) = output_channel_multiplier_shift(
                                params.input_quant,
                                params.weights_quant,
                                params.weights_per_channel_quant,
                                params.output_quant,
                                out_channel,
                            );
                            let mut acc = params
                                .bias
                                .map(|bias| bias[out_channel])
                                .unwrap_or_default();

                            for filter_y in 0..depthwise_dims.filter_h {
                                let in_y = out_y * stride_h + filter_y * dilation_h;
                                if in_y < pad_h || in_y >= input_h + pad_h {
                                    continue;
                                }
                                let in_y = in_y - pad_h;

                                for filter_x in 0..depthwise_dims.filter_w {
                                    let in_x = out_x * stride_w + filter_x * dilation_w;
                                    if in_x < pad_w || in_x >= input_w + pad_w {
                                        continue;
                                    }
                                    let in_x = in_x - pad_w;

                                    let input = params.input[nhwc_index(
                                        batch, in_y, in_x, in_channel, input_h, input_w, input_c,
                                    )] as i32
                                        - params.input_quant.zero_point;
                                    let weight = params.weights[depthwise_weight_index(
                                        filter_y,
                                        filter_x,
                                        in_channel,
                                        channel_multiplier,
                                        depthwise_dims,
                                    )] as i32
                                        - params.weights_quant.zero_point;
                                    acc = acc.saturating_add(input.saturating_mul(weight));
                                }
                            }

                            let scaled = requantize(acc, multiplier, shift, params.output_quant);
                            params.output[nhwc_index(
                                batch,
                                out_y,
                                out_x,
                                out_channel,
                                output_h,
                                output_w,
                                output_c,
                            )] = apply_activation(scaled, params.activation, params.output_quant);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn fully_connected(&mut self, params: FullyConnectedParams<'_>) -> Status {
        validate_len(params.output, params.output_depth)?;
        let [output_depth, input_depth] = params.weights_shape;
        if params.output_depth != output_depth
            || params.weights.len() != output_depth * input_depth
            || params.input.len() != input_depth
        {
            return Err(KernelError::InvalidShape);
        }
        validate_bias(params.bias, output_depth)?;

        for out_channel in 0..output_depth {
            let (multiplier, shift) = output_channel_multiplier_shift(
                params.input_quant,
                params.weights_quant,
                params.weights_per_channel_quant,
                params.output_quant,
                out_channel,
            );
            let mut acc = params
                .bias
                .map(|bias| bias[out_channel])
                .unwrap_or_default();
            for in_channel in 0..input_depth {
                let input = params.input[in_channel] as i32 - params.input_quant.zero_point;
                let weight = params.weights[out_channel * input_depth + in_channel] as i32
                    - params.weights_quant.zero_point;
                acc = acc.saturating_add(input.saturating_mul(weight));
            }

            let scaled = requantize(acc, multiplier, shift, params.output_quant);
            params.output[out_channel] =
                apply_activation(scaled, params.activation, params.output_quant);
        }

        Ok(())
    }

    fn avg_pool(&mut self, params: PoolParams<'_>) -> Status {
        pool(params, PoolKind::Average)
    }

    fn max_pool(&mut self, params: PoolParams<'_>) -> Status {
        pool(params, PoolKind::Max)
    }

    fn softmax(&mut self, params: SoftmaxParams<'_>) -> Status {
        let [batches, classes] = params.input_shape;
        if params.input.len() != batches * classes || params.output.len() != batches * classes {
            return Err(KernelError::InvalidShape);
        }

        let mut exps: Vec<f32> = vec![0.0; classes];
        for batch in 0..batches {
            let offset = batch * classes;
            let mut max_input = i8::MIN;
            for class in 0..classes {
                max_input = max_input.max(params.input[offset + class]);
            }

            let mut sum = 0.0f32;
            for (class, exp) in exps.iter_mut().enumerate() {
                let centered = (params.input[offset + class] as i32 - max_input as i32) as f32;
                let real = centered * params.input_quant.scale * params.beta;
                *exp = libm::expf(real);
                sum += *exp;
            }

            if sum == 0.0 {
                return Err(KernelError::InternalError);
            }

            for (class, exp) in exps.iter().enumerate() {
                let probability = *exp / sum;
                let quantized = round_f32_to_i32(probability / params.output_quant.scale)
                    + params.output_quant.zero_point;
                params.output[offset + class] = clamp_i8(quantized);
            }
        }

        Ok(())
    }

    fn add(&mut self, params: ElementwiseAddParams<'_>) -> Status {
        if params.input1.len() != params.input2.len() || params.output.len() != params.input1.len()
        {
            return Err(KernelError::InvalidShape);
        }

        for index in 0..params.output.len() {
            let lhs = (params.input1[index] as i32 - params.input1_quant.zero_point) as f32
                * params.input1_quant.scale;
            let rhs = (params.input2[index] as i32 - params.input2_quant.zero_point) as f32
                * params.input2_quant.scale;
            let quantized = round_f32_to_i32((lhs + rhs) / params.output_quant.scale)
                + params.output_quant.zero_point;
            params.output[index] =
                apply_activation(quantized, params.activation, params.output_quant);
        }

        Ok(())
    }

    fn mul(&mut self, params: MulParams<'_>) -> Status {
        // Element-wise when lengths match; otherwise the shorter operand is
        // broadcast per trailing (channel) dimension. Multiply is commutative,
        // so treat the longer operand as the dense one.
        let (dense, dense_q, per_ch, per_ch_q) = if params.input1.len() >= params.input2.len() {
            (
                params.input1,
                params.input1_quant,
                params.input2,
                params.input2_quant,
            )
        } else {
            (
                params.input2,
                params.input2_quant,
                params.input1,
                params.input1_quant,
            )
        };

        if per_ch.is_empty()
            || dense.len() % per_ch.len() != 0
            || params.output.len() != dense.len()
        {
            return Err(KernelError::InvalidShape);
        }

        let channels = per_ch.len();
        for index in 0..params.output.len() {
            let lhs = (dense[index] as i32 - dense_q.zero_point) as f32 * dense_q.scale;
            let rhs = (per_ch[index % channels] as i32 - per_ch_q.zero_point) as f32 * per_ch_q.scale;
            let quantized = round_f32_to_i32((lhs * rhs) / params.output_quant.scale)
                + params.output_quant.zero_point;
            params.output[index] =
                apply_activation(quantized, params.activation, params.output_quant);
        }

        Ok(())
    }
}

#[derive(Clone, Copy)]
enum PoolKind {
    Average,
    Max,
}

fn pool(params: PoolParams<'_>, kind: PoolKind) -> Status {
    validate_len(params.input, product(&params.input_shape))?;
    validate_len(params.output, product(&params.output_shape))?;

    let [batches, input_h, input_w, channels] = params.input_shape;
    let [output_batches, output_h, output_w, output_channels] = params.output_shape;
    if batches != output_batches || channels != output_channels {
        return Err(KernelError::InvalidShape);
    }

    let stride_h = positive_i32_to_usize(params.stride_h)?;
    let stride_w = positive_i32_to_usize(params.stride_w)?;
    let filter_h = positive_i32_to_usize(params.filter_h)?;
    let filter_w = positive_i32_to_usize(params.filter_w)?;
    let pad_h = compute_padding(input_h, filter_h, stride_h, params.padding);
    let pad_w = compute_padding(input_w, filter_w, stride_w, params.padding);
    let (multiplier, shift) =
        quantize_multiplier((params.input_quant.scale / params.output_quant.scale) as f64);

    for batch in 0..batches {
        for out_y in 0..output_h {
            for out_x in 0..output_w {
                for channel in 0..channels {
                    let mut acc = 0i32;
                    let mut count = 0i32;
                    let mut max_value = i8::MIN;

                    for filter_y in 0..filter_h {
                        let in_y = out_y * stride_h + filter_y;
                        if in_y < pad_h || in_y >= input_h + pad_h {
                            continue;
                        }
                        let in_y = in_y - pad_h;

                        for filter_x in 0..filter_w {
                            let in_x = out_x * stride_w + filter_x;
                            if in_x < pad_w || in_x >= input_w + pad_w {
                                continue;
                            }
                            let in_x = in_x - pad_w;
                            let input = params.input[nhwc_index(
                                batch, in_y, in_x, channel, input_h, input_w, channels,
                            )];
                            acc += input as i32 - params.input_quant.zero_point;
                            count += 1;
                            max_value = max_value.max(input);
                        }
                    }

                    if count == 0 {
                        return Err(KernelError::InvalidShape);
                    }

                    let quantized = match kind {
                        PoolKind::Average => {
                            let average = round_divide(acc, count);
                            requantize(average, multiplier, shift, params.output_quant)
                        }
                        PoolKind::Max => {
                            let centered = max_value as i32 - params.input_quant.zero_point;
                            requantize(centered, multiplier, shift, params.output_quant)
                        }
                    };
                    params.output
                        [nhwc_index(batch, out_y, out_x, channel, output_h, output_w, channels)] =
                        apply_activation(quantized, params.activation, params.output_quant);
                }
            }
        }
    }

    Ok(())
}

fn validate_len<T>(slice: &[T], expected: usize) -> Status {
    if slice.len() == expected {
        Ok(())
    } else {
        Err(KernelError::InvalidShape)
    }
}

fn validate_bias(bias: Option<&[i32]>, expected: usize) -> Status {
    match bias {
        Some(bias) => validate_len(bias, expected),
        None => Ok(()),
    }
}

fn product<const N: usize>(shape: &[usize; N]) -> usize {
    shape.iter().product()
}

fn positive_i32_to_usize(value: i32) -> Result<usize, KernelError> {
    if value > 0 {
        Ok(value as usize)
    } else {
        Err(KernelError::InvalidShape)
    }
}

fn effective_filter_size(filter_size: usize, dilation: usize) -> usize {
    (filter_size - 1) * dilation + 1
}

fn nhwc_index(
    batch: usize,
    y: usize,
    x: usize,
    channel: usize,
    height: usize,
    width: usize,
    channels: usize,
) -> usize {
    ((batch * height + y) * width + x) * channels + channel
}

fn conv_weight_index(
    output_channel: usize,
    filter_y: usize,
    filter_x: usize,
    input_channel: usize,
    filter_h: usize,
    filter_w: usize,
    input_channels: usize,
) -> usize {
    ((output_channel * filter_h + filter_y) * filter_w + filter_x) * input_channels + input_channel
}

fn depthwise_weight_index(
    filter_y: usize,
    filter_x: usize,
    input_channel: usize,
    channel_multiplier: usize,
    dims: DepthwiseDims,
) -> usize {
    let output_channel = input_channel * dims.depth_multiplier + channel_multiplier;
    if dims.tflite_layout {
        (filter_y * dims.filter_w + filter_x) * (dims.input_channels * dims.depth_multiplier)
            + output_channel
    } else {
        ((filter_y * dims.filter_w + filter_x) * dims.input_channels + input_channel)
            * dims.depth_multiplier
            + channel_multiplier
    }
}

#[derive(Clone, Copy)]
struct DepthwiseDims {
    filter_h: usize,
    filter_w: usize,
    input_channels: usize,
    depth_multiplier: usize,
    tflite_layout: bool,
}

fn depthwise_filter_dims(
    weights_shape: [usize; 4],
    input_channels: usize,
    depth_multiplier: usize,
) -> Result<DepthwiseDims, KernelError> {
    if weights_shape[0] == 1 {
        if input_channels == 0 {
            return Err(KernelError::InvalidShape);
        }
        Ok(DepthwiseDims {
            filter_h: weights_shape[1],
            filter_w: weights_shape[2],
            input_channels,
            depth_multiplier: weights_shape[3] / input_channels,
            tflite_layout: true,
        })
    } else {
        Ok(DepthwiseDims {
            filter_h: weights_shape[0],
            filter_w: weights_shape[1],
            input_channels: weights_shape[2],
            depth_multiplier: weights_shape[3],
            tflite_layout: false,
        })
    }
    .and_then(|dims| {
        if dims.input_channels == input_channels && dims.depth_multiplier == depth_multiplier {
            Ok(dims)
        } else {
            Err(KernelError::InvalidShape)
        }
    })
}

fn multiply_by_quantized_multiplier(x: i32, multiplier: i32, shift: i32) -> i32 {
    let total_shift = 31 - shift;
    if total_shift <= 0 {
        return saturating_left_shift(x.saturating_mul(multiplier), (-total_shift) as u32);
    }
    let round = 1i64 << (total_shift - 1);
    (((x as i64 * multiplier as i64) + round) >> total_shift) as i32
}

fn saturating_left_shift(value: i32, shift: u32) -> i32 {
    if value == 0 {
        return 0;
    }

    if shift >= 31 {
        if value >= 0 {
            i32::MAX
        } else {
            i32::MIN
        }
    } else {
        ((value as i64) << shift).clamp(i32::MIN as i64, i32::MAX as i64) as i32
    }
}

fn quantize_multiplier(scale: f64) -> (i32, i32) {
    if scale <= 0.0 {
        return (0, 0);
    }

    let mut significand = scale;
    let mut shift = 0i32;

    while significand < 0.5 {
        significand *= 2.0;
        shift -= 1;
    }
    while significand >= 1.0 {
        significand /= 2.0;
        shift += 1;
    }

    let mut q = libm::round(significand * (1i64 << 31) as f64) as i64;
    if q == 1i64 << 31 {
        q /= 2;
        shift += 1;
    }

    (q as i32, shift)
}

fn output_channel_multiplier_shift(
    input_quant: QuantParam,
    weights_quant: QuantParam,
    weights_per_channel_quant: Option<PerChannelQuantParam<'_>>,
    output_quant: QuantParam,
    output_channel: usize,
) -> (i32, i32) {
    let weight_scale = weights_per_channel_quant
        .and_then(|per_channel| per_channel.scales.get(output_channel).copied())
        .unwrap_or(weights_quant.scale);
    quantize_multiplier((input_quant.scale * weight_scale / output_quant.scale) as f64)
}

fn requantize(acc: i32, multiplier: i32, shift: i32, output_quant: QuantParam) -> i32 {
    multiply_by_quantized_multiplier(acc, multiplier, shift) + output_quant.zero_point
}

fn apply_activation(val: i32, activation: FusedActivation, output_quant: QuantParam) -> i8 {
    let min = match activation {
        FusedActivation::None | FusedActivation::Sigmoid | FusedActivation::SignBit => {
            i8::MIN as i32
        }
        FusedActivation::Relu | FusedActivation::Relu6 => {
            (i8::MIN as i32).max(output_quant.zero_point)
        }
        FusedActivation::ReluN1To1 | FusedActivation::Tanh => (i8::MIN as i32)
            .max(output_quant.zero_point + round_f32_to_i32(-1.0 / output_quant.scale)),
    };
    let max = match activation {
        FusedActivation::Relu6 => (i8::MAX as i32)
            .min(output_quant.zero_point + round_f32_to_i32(6.0 / output_quant.scale)),
        FusedActivation::ReluN1To1 | FusedActivation::Tanh | FusedActivation::Sigmoid => (i8::MAX
            as i32)
            .min(output_quant.zero_point + round_f32_to_i32(1.0 / output_quant.scale)),
        FusedActivation::None | FusedActivation::Relu | FusedActivation::SignBit => i8::MAX as i32,
    };

    clamp_i8(val.clamp(min, max))
}

fn clamp_i8(value: i32) -> i8 {
    value.clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

fn compute_padding(
    input_size: usize,
    filter_size: usize,
    stride: usize,
    padding: Padding,
) -> usize {
    match padding {
        Padding::Valid => 0,
        Padding::Same => {
            let out_size = input_size.div_ceil(stride);
            let pad = ((out_size - 1) * stride + filter_size).saturating_sub(input_size);
            pad / 2
        }
    }
}

fn round_f32_to_i32(value: f32) -> i32 {
    libm::roundf(value) as i32
}

fn round_divide(numerator: i32, denominator: i32) -> i32 {
    if numerator >= 0 {
        (numerator + denominator / 2) / denominator
    } else {
        (numerator - denominator / 2) / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNIT_QUANT: QuantParam = QuantParam {
        scale: 1.0,
        zero_point: 0,
    };

    #[test]
    fn fully_connected_identity_scale() {
        let mut backend = RefBackend;
        let input = [2, -3];
        let weights = [4, 5, -1, 6];
        let mut output = [0; 2];

        backend
            .fully_connected(FullyConnectedParams {
                input: &input,
                input_quant: UNIT_QUANT,
                weights: &weights,
                weights_shape: [2, 2],
                weights_quant: UNIT_QUANT,
                weights_per_channel_quant: None,
                bias: Some(&[1, -2]),
                output: &mut output,
                output_depth: 2,
                output_quant: UNIT_QUANT,
                activation: FusedActivation::None,
            })
            .unwrap();

        assert_eq!(output, [-6, -22]);
    }

    #[test]
    fn add_identity_scale() {
        let mut backend = RefBackend;
        let input1 = [1, -2, 3];
        let input2 = [4, 5, -6];
        let mut output = [0; 3];

        backend
            .add(ElementwiseAddParams {
                input1: &input1,
                input1_quant: UNIT_QUANT,
                input2: &input2,
                input2_quant: UNIT_QUANT,
                output: &mut output,
                output_quant: UNIT_QUANT,
                activation: FusedActivation::None,
            })
            .unwrap();

        assert_eq!(output, [5, 3, -3]);
    }

    #[test]
    fn avg_pool_valid() {
        let mut backend = RefBackend;
        let input = [1, 3, 5, 7];
        let mut output = [0; 1];

        backend
            .avg_pool(PoolParams {
                input: &input,
                input_shape: [1, 2, 2, 1],
                input_quant: UNIT_QUANT,
                output: &mut output,
                output_shape: [1, 1, 1, 1],
                output_quant: UNIT_QUANT,
                stride_w: 1,
                stride_h: 1,
                filter_w: 2,
                filter_h: 2,
                padding: Padding::Valid,
                activation: FusedActivation::None,
            })
            .unwrap();

        assert_eq!(output, [4]);
    }

    #[test]
    fn conv2d_single_filter_valid() {
        let mut backend = RefBackend;
        let input = [1, 2, 3, 4];
        let weights = [1, 0, 0, 1];
        let mut output = [0; 1];

        backend
            .conv2d(Conv2dParams {
                input: &input,
                input_shape: [1, 2, 2, 1],
                input_quant: UNIT_QUANT,
                weights: &weights,
                weights_shape: [1, 2, 2, 1],
                weights_quant: UNIT_QUANT,
                weights_per_channel_quant: None,
                bias: None,
                output: &mut output,
                output_shape: [1, 1, 1, 1],
                output_quant: UNIT_QUANT,
                stride_w: 1,
                stride_h: 1,
                dilation_w_factor: 1,
                dilation_h_factor: 1,
                padding: Padding::Valid,
                activation: FusedActivation::None,
                scratch: &mut [],
            })
            .unwrap();

        assert_eq!(output, [5]);
    }

    #[test]
    fn depthwise_accepts_tflite_filter_layout() {
        let mut backend = RefBackend;
        let input = [1, 2, 3, 4];
        let weights = [1, 0, 0, 1];
        let mut output = [0; 1];

        backend
            .depthwise_conv2d(DepthwiseConv2dParams {
                input: &input,
                input_shape: [1, 2, 2, 1],
                input_quant: UNIT_QUANT,
                weights: &weights,
                weights_shape: [1, 2, 2, 1],
                weights_quant: UNIT_QUANT,
                weights_per_channel_quant: None,
                bias: None,
                output: &mut output,
                output_shape: [1, 1, 1, 1],
                output_quant: UNIT_QUANT,
                stride_w: 1,
                stride_h: 1,
                dilation_w_factor: 1,
                dilation_h_factor: 1,
                depth_multiplier: 1,
                padding: Padding::Valid,
                activation: FusedActivation::None,
                scratch: &mut [],
            })
            .unwrap();

        assert_eq!(output, [5]);
    }

    #[test]
    fn softmax_outputs_probability_distribution() {
        let mut backend = RefBackend;
        let input = [0, 0];
        let mut output = [0; 2];

        backend
            .softmax(SoftmaxParams {
                input: &input,
                input_shape: [1, 2],
                input_quant: QuantParam {
                    scale: 1.0,
                    zero_point: 0,
                },
                output: &mut output,
                output_quant: QuantParam {
                    scale: 1.0 / 256.0,
                    zero_point: -128,
                },
                beta: 1.0,
                scratch: &mut [],
            })
            .unwrap();

        assert_eq!(output, [0, 0]);
    }
}
