pub mod buffer_iterator;
pub mod flat_index_generator;
pub mod flat_iterator;
pub mod tensor_iterator;

pub(super) mod collapse_contiguous;
mod util;

pub use flat_iterator::*;
