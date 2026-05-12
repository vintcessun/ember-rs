use ember_infer_macros::model;
use ember_infer_ref::RefBackend;

extern crate self as microflow;
extern crate self as nalgebra;

pub mod buffer {
    pub type Buffer2D<T, const ROWS: usize, const COLS: usize> = [[T; COLS]; ROWS];
    pub type Buffer4D<
        T,
        const BATCH: usize,
        const HEIGHT: usize,
        const WIDTH: usize,
        const CHANNELS: usize,
    > = [[[[T; CHANNELS]; WIDTH]; HEIGHT]; BATCH];
}

#[macro_export]
macro_rules! matrix {
    ($( $($value:expr),+ );+ $(;)?) => {
        [$( [ $($value),+ ] ),+]
    };
}

#[allow(unused_imports)]
#[path = "speech.rs"]
mod speech_samples;

#[allow(unused_imports)]
#[path = "person_detect.rs"]
mod person_detect_samples;

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

fn assert_top_class(model: &str, sample: &str, scores: &[f32], expected: usize) {
    let (actual, actual_score) = scores
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .unwrap();

    assert_eq!(
        actual, expected,
        "{model} {sample} expected class {expected}, got class {actual}; scores={scores:?}"
    );
    assert!(
        actual_score > 0.5,
        "{model} {sample} expected class {expected} confidence > 0.5, got {actual_score}; scores={scores:?}"
    );
}

fn run_speech_sample(name: &str, input: &[i8]) -> Vec<f32> {
    assert_eq!(input.len(), SpeechModel::input_len());

    let mut backend = RefBackend;
    let mut output = vec![0i8; SpeechModel::output_len()];

    SpeechModel::predict_quantized(&mut backend, input, &mut output).unwrap();

    let scores: Vec<f32> = output
        .iter()
        .map(|&v| {
            dequantize(
                v,
                SpeechModel::output_scale(),
                SpeechModel::output_zero_point(),
            )
        })
        .collect();

    println!("[speech:{name}] input length = {}", input.len());
    println!("[speech:{name}] output length = {} classes", output.len());
    for (i, score) in scores.iter().enumerate() {
        println!("[speech:{name}]   class {i} score = {score:.4}");
    }
    let sum: f32 = scores.iter().sum();
    println!("[speech:{name}] sum of scores = {sum:.4} (should be ~= 1.0)");

    assert_eq!(output.len(), SpeechModel::output_len());
    for (index, score) in scores.iter().copied().enumerate() {
        assert!(
            (0.0..=1.0).contains(&score),
            "speech {name} output[{index}] out of range: score={score}"
        );
    }
    assert!(
        (sum - 1.0).abs() < 0.05,
        "speech {name} scores should sum to ~= 1.0, got {sum}"
    );

    scores
}

fn run_person_detect_sample(name: &str, input: &[i8]) -> Vec<f32> {
    assert_eq!(input.len(), PersonDetectModel::input_len());

    let mut backend = RefBackend;
    let mut output = vec![0i8; PersonDetectModel::output_len()];

    PersonDetectModel::predict_quantized(&mut backend, input, &mut output).unwrap();

    println!(
        "[person_detect:{name}] input length = {} pixels (i8)",
        input.len()
    );
    println!(
        "[person_detect:{name}] output length = {} classes",
        output.len()
    );

    let scores: Vec<f32> = output
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let label = match i {
                0 => "no_person",
                1 => "person",
                _ => "unknown",
            };
            let score = dequantize(
                v,
                PersonDetectModel::output_scale(),
                PersonDetectModel::output_zero_point(),
            );
            println!(
                "[person_detect:{name}]   class {i} ({label}) raw i8 = {v}  ->  score = {score:.4}"
            );
            score
        })
        .collect();

    assert_eq!(output.len(), PersonDetectModel::output_len());
    for (index, score) in scores.iter().copied().enumerate() {
        assert!(
            (0.0..=1.0).contains(&score),
            "person_detect {name} output[{index}] out of range: score={score}"
        );
    }

    scores
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
    println!(
        "[sine] input x = {:.4}  ->  quantized input = {}",
        x, q_input
    );
    println!(
        "[sine] raw output i8 = {}  ->  dequantized f32 = {:.4}",
        output[0], result
    );
    println!(
        "[sine] expected sin({:.4}) = {:.4}  |  error = {:.4}  |  tolerance = 0.10  {}",
        x,
        expected,
        (result - expected).abs(),
        if (result - expected).abs() < 0.1 {
            "✓"
        } else {
            "✗"
        }
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
    let yes_scores = run_speech_sample("yes", speech_samples::YES.as_flattened());
    let no_scores = run_speech_sample("no", speech_samples::NO.as_flattened());

    assert_top_class("speech", "yes", &yes_scores, 2);
    assert_top_class("speech", "no", &no_scores, 3);
    assert_ne!(
        yes_scores, no_scores,
        "speech samples should produce different outputs"
    );
}

#[test]
fn person_detect_compiles_and_runs() {
    let person_scores = run_person_detect_sample(
        "person",
        person_detect_samples::PERSON
            .as_flattened()
            .as_flattened()
            .as_flattened(),
    );
    let no_person_scores = run_person_detect_sample(
        "no_person",
        person_detect_samples::NO_PERSON
            .as_flattened()
            .as_flattened()
            .as_flattened(),
    );

    assert_top_class("person_detect", "person", &person_scores, 1);
    assert_top_class("person_detect", "no_person", &no_person_scores, 1);
    assert_ne!(
        person_scores, no_person_scores,
        "person_detect samples should produce different outputs"
    );
}
