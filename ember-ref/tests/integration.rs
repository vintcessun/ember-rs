use ember_macros::model;
use ember_ref::RefBackend;

#[model("models/sine.tflite")]
struct SineModel;

#[model("models/speech.tflite")]
struct SpeechModel;

#[model("models/person_detect.tflite")]
struct PersonDetectModel;

fn quantize(val: f32, scale: f32, zero_point: i32) -> i8 {
    ((val / scale) + zero_point as f32)
        .round()
        .clamp(i8::MIN as f32, i8::MAX as f32) as i8
}

fn dequantize(val: i8, scale: f32, zero_point: i32) -> f32 {
    (val as i32 - zero_point) as f32 * scale
}

fn assert_sine_close(x: f32, expected: f32) {
    let mut backend = RefBackend;
    let q_input = quantize(x, SineModel::input_scale(), SineModel::input_zero_point());
    let input = [q_input; SineModel::input_len()];
    let mut output = [0i8; SineModel::output_len()];

    SineModel::predict_quantized(&mut backend, &input, &mut output).unwrap();

    let result = dequantize(
        output[0],
        SineModel::output_scale(),
        SineModel::output_zero_point(),
    );
    assert!(
        (result - expected).abs() < 0.1,
        "expected sin({x}) ~= {expected}, got {result}; q_input={q_input}, q_output={:?}",
        output
    );
}

#[test]
fn sine_end_to_end_zero() {
    assert_sine_close(0.0, 0.0);
}

#[test]
fn sine_end_to_end_pi_over_2() {
    assert_sine_close(core::f32::consts::FRAC_PI_2, 1.0);
}

#[test]
fn sine_end_to_end_pi() {
    assert_sine_close(core::f32::consts::PI, 0.0);
}

#[test]
fn speech_compiles_and_runs() {
    let mut backend = RefBackend;
    let input = vec![0i8; SpeechModel::input_len()];
    let mut output = vec![0i8; SpeechModel::output_len()];

    SpeechModel::predict_quantized(&mut backend, &input, &mut output).unwrap();

    assert_eq!(output.len(), SpeechModel::output_len());
    for (index, value) in output.iter().copied().enumerate() {
        let dequantized = dequantize(
            value,
            SpeechModel::output_scale(),
            SpeechModel::output_zero_point(),
        );
        assert!(
            (-1.0..=1.0).contains(&dequantized),
            "speech output[{index}] out of range: q={value}, dequantized={dequantized}"
        );
    }
}

#[test]
fn person_detect_compiles_and_runs() {
    let mut backend = RefBackend;
    let input = vec![0i8; PersonDetectModel::input_len()];
    let mut output = vec![0i8; PersonDetectModel::output_len()];

    PersonDetectModel::predict_quantized(&mut backend, &input, &mut output).unwrap();

    assert_eq!(output.len(), PersonDetectModel::output_len());
}
