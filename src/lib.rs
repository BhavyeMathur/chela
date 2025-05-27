#![allow(clippy::needless_lifetimes)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_map)]
#![allow(clippy::new_ret_no_self)]

pub mod tensor;
pub use tensor::*;

pub mod linalg;
pub use linalg::*;

pub mod util;
pub use util::*;

#[cfg(use_apple_accelerate)]
mod accelerate;

pub mod autograd;
pub use autograd::*;
