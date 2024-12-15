mod indexer_impl;
pub(crate) mod indexer;

pub mod index;
#[macro_use]
pub mod slice_index;

pub struct Axis(pub usize);
