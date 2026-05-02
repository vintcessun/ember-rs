# ember-rs

`ember-rs` is a `no_std` embedded TinyML inference engine for INT8 models.
It is a fork and redesign of [microflow-rs](https://github.com/matteocarnelos/microflow-rs),
with a pluggable backend interface so optimized hardware kernels can be swapped in without
changing model-facing code.

The long-term goal is to be the Burn-style backend abstraction for embedded INT8 inference:
the model graph is generated at compile time, while operator execution is delegated to a
backend that implements `ember_core::KernelBackend`.

## Workspace

This repository currently contains three crates:

| Crate | Purpose |
|---|---|
| `ember-core` | Core `no_std` API: `KernelBackend`, operator parameter structs, errors, and status type. |
| `ember-ref` | Pure Rust reference backend crate. It currently provides a compiling `RefBackend` stub. |
| `ember-macros` | Procedural macro crate forked from microflow. Backend-aware generation is still being migrated. |

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

## Latest Usage

At the current stage, the stable contract is the low-level backend interface in
`ember-core`. Applications and generated code call a concrete backend through static
dispatch:

```rust
use ember_core::{
    Conv2dParams, ElementwiseAddParams, FullyConnectedParams, KernelBackend, PoolParams,
    SoftmaxParams, Status,
};

fn run_model<B: KernelBackend>(backend: &mut B) -> Status {
    // Generated code from ember-macros will build these parameter structs
    // from compile-time model metadata and tensor buffers.
    //
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

`ember-ref` can be used today as a placeholder backend while the reference kernels are
ported:

```rust
use ember_ref::RefBackend;

let mut backend = RefBackend;
```

`RefBackend` currently returns `KernelError::InternalError` for every operator. It exists
so the workspace and downstream backend integration can compile while the pure Rust
operator implementations are migrated.

## Custom Backends

To add a backend, implement the `ember_core::KernelBackend` trait for your backend type.
The trait is the only required backend contract.

```rust
use ember_core::{
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

Parameter structs in `ember-core` intentionally mirror TFLite Micro naming and layout
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
array sizing are intended to be handled at compile time by `ember-macros`.

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
