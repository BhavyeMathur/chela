#![allow(clippy::needless_lifetimes)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_map)]
#![allow(clippy::new_ret_no_self)]
#![allow(clippy::erasing_op)]
#![allow(clippy::identity_op)]
#![allow(ambiguous_glob_reexports)]

#[macro_use]
pub mod macros;

pub mod acceleration;

pub mod ndarray;
pub use ndarray::*;

pub mod tensor;
pub use tensor::*;

pub mod common;
pub use common::*;

pub mod linalg;
pub use linalg::*;

pub mod util;
pub use util::*;

pub mod autograd;
pub use autograd::*;

pub mod ops;
