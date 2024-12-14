pub(crate) mod indexer;
pub(crate) mod indexer_impl;

pub mod index;
#[macro_use]
pub mod slice_index;

pub struct Axis(pub usize);
