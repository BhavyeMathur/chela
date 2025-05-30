#![allow(clippy::needless_lifetimes)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_map)]
#![allow(clippy::new_ret_no_self)]
#![allow(ambiguous_glob_reexports)]


#[cfg(apple_accelerate)]
mod accelerate;

pub mod ndarray;
pub use ndarray::*;

pub mod tensor;
pub use tensor::*;

pub mod traits;
pub use traits::*;

pub mod linalg;
pub use linalg::*;

pub mod util;
pub use util::*;

pub mod autograd;
pub use autograd::*;
