# ember-rs

[![CI](https://github.com/vintcessun/ember-rs/actions/workflows/cargo.yml/badge.svg)](https://github.com/vintcessun/ember-rs/actions/workflows/cargo.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

`ember-rs` is a `no_std` embedded TinyML inference engine for INT8 models.
It is a fork and redesign of [microflow-rs](https://github.com/matteocarnelos/microflow-rs),
with a pluggable backend interface so optimized hardware kernels can be swapped in without
changing model-facing code.

The long-term goal is to be the Burn-style backend abstraction for embedded INT8 inference:
the model graph is generated at compile time, while operator execution is delegated to a
backend that implements `ember_infer_core::KernelBackend`.

## Workspace

This repository currently contains three crates:

| Crate | Purpose |
|---|---|
| `ember-infer-core` | Core `no_std` API: `KernelBackend`, operator parameter structs, errors, and status type. |
| `ember-infer-ref` | Pure Rust reference backend. Implements all 7 operators (`conv2d`, `depthwise_conv2d`, `fully_connected`, `avg_pool`, `max_pool`, `softmax`, `add`) with correct INT8 fixed-point quantization arithmetic. Verified against `sine.tflite`, `speech.tflite`, and `person_detect.tflite`. |
| `ember-infer-macros` | Procedural macro crate that reads `.tflite` models and generates backend-dispatched inference wrappers. |

The ESP32-S3 backend is intentionally not part of this workspace. It lives in a separate
repository as `ember-esp` and implements the same `KernelBackend` trait using Espressif
`esp-nn` kernels.

## Requirements

Use nightly Rust:

```bash
cargo +nightly check --workspace
```

The workspace includes `rust-toolchain.toml`, so normal `cargo` commands should select
nightly automatically in this directory.

## Crates

For an application using the generated model wrapper with the reference backend:

```toml
[dependencies]
ember-infer-core = "0.1.0"
ember-infer-macros = "0.1.0"
ember-infer-ref = "0.1.0"
```

Rust imports use underscores, because Cargo package names with hyphens are exposed as
crate names with underscores:

```rust
use ember_infer_macros::model;
use ember_infer_ref::RefBackend;
```

## Usage: Model Inference

ember-rs is designed so application code references a TensorFlow Lite model at compile
time, chooses a concrete backend, and then runs inference through static dispatch.

The intended high-level flow is:

1. Put a quantized INT8 `.tflite` model in your project, for example
   `models/sine.tflite`.
2. Annotate a model struct with `ember-infer-macros`.
3. Create a backend value, such as `RefBackend` or an external `EspBackend`.
4. Pass input and output buffers to the generated inference method.

The macro generates `input_len()`, `output_len()`, `scratch_len::<B>()`,
`predict_quantized(...)`, and `predict_quantized_with_scratch(...)` for the annotated
struct:

```rust
use ember_infer_macros::model;
use ember_infer_ref::RefBackend;

#[model("models/sine.tflite")]
pub struct SineModel;

fn main() -> Result<(), ember_infer_core::KernelError> {
    let mut backend = RefBackend;

    let input = [0i8; SineModel::input_len()];
    let mut output = [0i8; SineModel::output_len()];

    SineModel::predict_quantized(&mut backend, &input, &mut output)?;

    Ok(())
}
```

The important part is that the backend is a normal argument. Switching inference engines
does not change the model wrapper:

```rust,ignore
use ember_esp::EspBackend;

let mut backend = EspBackend::new();
let input = [0i8; SineModel::input_len()];
let mut output = [0i8; SineModel::output_len()];

// Pick a fixed size appropriate for your model/backend, or derive it from
// `SineModel::scratch_len::<EspBackend>()` during bring-up.
const SCRATCH_LEN: usize = 4096;
let mut scratch = [0u8; SCRATCH_LEN];

SineModel::predict_quantized_with_scratch(&mut backend, &input, &mut output, &mut scratch)?;
```

For backends that need scratch memory, query the required length for that backend:

```rust,ignore
let required = SineModel::scratch_len::<EspBackend>();
```

On embedded targets you usually turn that value into a fixed stack/static buffer according
to your platform's memory policy.

The generated inference methods are generic over the backend:

```rust
pub fn predict_quantized<B: ember_infer_core::KernelBackend>(
    backend: &mut B,
    input: &[i8],
    output: &mut [i8],
) -> ember_infer_core::Status

pub fn predict_quantized_with_scratch<B: ember_infer_core::KernelBackend>(
    backend: &mut B,
    input: &[i8],
    output: &mut [i8],
    scratch: &mut [u8],
) -> ember_infer_core::Status
```

`input` and `output` must match `input_len()` and `output_len()`. If either slice has the
wrong length, inference returns `KernelError::InvalidShape`.

The current generated API is quantized-only. Feed INT8 input tensors and read INT8 output
tensors. Floating-point convenience helpers can be added above this API by quantizing into
an INT8 input buffer before calling `predict_quantized`.

### Generated Operator Calls

`ember-infer-macros` turns each supported TFLite operator into a `KernelBackend` call. In other
words, generated model code is equivalent to this low-level pattern:

```rust
use ember_infer_core::{
    Conv2dParams, ElementwiseAddParams, FullyConnectedParams, KernelBackend, PoolParams,
    SoftmaxParams, Status,
};

fn run_model<B: KernelBackend>(backend: &mut B) -> Status {
    // The macro emits concrete parameter structs using model metadata,
    // embedded weights, input/output slices, and intermediate buffers.
    // backend.conv2d(Conv2dParams { ... })?;
    // backend.fully_connected(FullyConnectedParams { ... })?;
    // backend.softmax(SoftmaxParams { ... })?;

    let _ = (
        core::mem::size_of::<Conv2dParams<'_>>(),
        core::mem::size_of::<FullyConnectedParams<'_>>(),
        core::mem::size_of::<PoolParams<'_>>(),
        core::mem::size_of::<SoftmaxParams<'_>>(),
        core::mem::size_of::<ElementwiseAddParams<'_>>(),
    );

    Ok(())
}
```

`ember-infer-ref` provides a complete pure-Rust INT8 reference implementation. Use it for
host-side testing, CI, and as the baseline when bringing up a new hardware backend:

```rust
use ember_infer_ref::RefBackend;

let mut backend = RefBackend;
SineModel::predict_quantized(&mut backend, &input, &mut output)?;
```

## Custom Backends

To add a backend, implement the `ember_infer_core::KernelBackend` trait for your backend type.
The trait is the only required backend contract.

```rust
use ember_infer_core::{
    Conv2dParams, DepthwiseConv2dParams, ElementwiseAddParams, FullyConnectedParams,
    KernelBackend, KernelError, PoolParams, SoftmaxParams, Status,
};

pub struct MyBackend;

impl KernelBackend for MyBackend {
    fn conv2d(&mut self, params: Conv2dParams<'_>) -> Status {
        let _ = params;
        Err(KernelError::InternalError)
    }

    fn depthwise_conv2d(&mut self, params: DepthwiseConv2dParams<'_>) -> Status {
        let _ = params;
        Err(KernelError::InternalError)
    }

    fn fully_connected(&mut self, params: FullyConnectedParams<'_>) -> Status {
        let _ = params;
        Err(KernelError::InternalError)
    }

    fn avg_pool(&mut self, params: PoolParams<'_>) -> Status {
        let _ = params;
        Err(KernelError::InternalError)
    }

    fn max_pool(&mut self, params: PoolParams<'_>) -> Status {
        let _ = params;
        Err(KernelError::InternalError)
    }

    fn softmax(&mut self, params: SoftmaxParams<'_>) -> Status {
        let _ = params;
        Err(KernelError::InternalError)
    }

    fn add(&mut self, params: ElementwiseAddParams<'_>) -> Status {
        let _ = params;
        Err(KernelError::InternalError)
    }
}
```

The required invoke methods are:

| Method | Operator |
|---|---|
| `conv2d` | `CONV_2D` |
| `depthwise_conv2d` | `DEPTHWISE_CONV_2D` |
| `fully_connected` | `FULLY_CONNECTED` |
| `avg_pool` | `AVERAGE_POOL_2D` |
| `max_pool` | `MAX_POOL_2D` |
| `softmax` | `SOFTMAX` |
| `add` | `ADD` |

Backends that need temporary memory should also override the scratch-size associated
functions:

```rust
impl KernelBackend for MyBackend {
    // required invoke methods omitted

    fn conv2d_scratch_size(
        input_shape: [usize; 4],
        weights_shape: [usize; 4],
        output_shape: [usize; 4],
    ) -> usize {
        let _ = (input_shape, weights_shape, output_shape);
        0
    }

    fn depthwise_conv2d_scratch_size(
        input_shape: [usize; 4],
        weights_shape: [usize; 4],
        output_shape: [usize; 4],
    ) -> usize {
        let _ = (input_shape, weights_shape, output_shape);
        0
    }

    fn softmax_scratch_size(num_classes: usize) -> usize {
        let _ = num_classes;
        0
    }
}
```

These functions default to `0`, which is appropriate for backends that do not need scratch
memory. Optimized kernels such as `esp-nn` or CMSIS-NN-style implementations should return
the exact number of bytes needed by the corresponding operator.

## Backend Semantics

Parameter structs in `ember-infer-core` intentionally mirror TFLite Micro naming and layout
semantics. Tensor data is INT8 and operator tensors use the same layouts expected by the
trait documentation:

| Parameter type | Layout |
|---|---|
| `Conv2dParams` input/output | NHWC |
| `Conv2dParams` weights | `[C_out, KH, KW, C_in]` |
| `DepthwiseConv2dParams` input/output | NHWC |
| `FullyConnectedParams` weights | `[output_depth, input_depth]` |
| `SoftmaxParams` input | `[batch, num_classes]` |

The trait covers the invoke phase only. Shape inference, tensor allocation, and scratch
array sizing are intended to be handled at compile time by `ember-infer-macros`.

## Development

Useful checks:

```bash
cargo +nightly fmt --all
cargo +nightly check --workspace
cargo +nightly clippy --workspace -- -D warnings
cargo +nightly doc --workspace --no-deps
```

## Lineage

`ember-rs` is based on `microflow-rs`, originally developed by Matteo Carnelos as part of
his master's thesis project at the University of Padova in collaboration with Grepit AB.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
