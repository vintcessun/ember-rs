#![no_std]

extern crate alloc;

use ember_core::{
    Conv2dParams, DepthwiseConv2dParams, ElementwiseAddParams, FullyConnectedParams, KernelBackend,
    KernelError, PoolParams, SoftmaxParams, Status,
};

/// Pure Rust reference implementation of [`KernelBackend`].
///
/// Ported from microflow-rs. Used for CI testing and platforms
/// where no hardware-accelerated backend is available.
pub struct RefBackend;

impl KernelBackend for RefBackend {
    fn conv2d(&mut self, _params: Conv2dParams<'_>) -> Status {
        Err(KernelError::InternalError)
    }

    fn depthwise_conv2d(&mut self, _params: DepthwiseConv2dParams<'_>) -> Status {
        Err(KernelError::InternalError)
    }

    fn fully_connected(&mut self, _params: FullyConnectedParams<'_>) -> Status {
        Err(KernelError::InternalError)
    }

    fn avg_pool(&mut self, _params: PoolParams<'_>) -> Status {
        Err(KernelError::InternalError)
    }

    fn max_pool(&mut self, _params: PoolParams<'_>) -> Status {
        Err(KernelError::InternalError)
    }

    fn softmax(&mut self, _params: SoftmaxParams<'_>) -> Status {
        Err(KernelError::InternalError)
    }

    fn add(&mut self, _params: ElementwiseAddParams<'_>) -> Status {
        Err(KernelError::InternalError)
    }
}
