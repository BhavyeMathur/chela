#[macro_use]
mod macros;

pub(crate) mod flatten;
pub(crate) mod homogenous;
pub(crate) mod nested;
pub(crate) mod shape;
pub(crate) mod haslength;
pub(crate) mod to_vec;
pub(crate) mod functions;

pub mod index;
pub use index::*;

pub mod axis;
pub use axis::*;
