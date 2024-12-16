use crate::dtype::RawDataType;

pub mod buffer_iterator;
pub(super) mod collapse_contiguous;
pub mod flat_index_generator;
pub mod flat_iterator;

pub use flat_iterator::*;
