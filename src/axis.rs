mod indexer_impl;
pub(crate) mod indexer;

pub mod index;
#[macro_use]
pub mod slice_index;
pub mod axes_traits;

pub struct Axis(pub usize);
