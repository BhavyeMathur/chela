mod indexer_impl;
pub(crate) mod indexer;

pub mod index;
#[macro_use]
pub mod slice_index;

pub struct Axis(pub usize);

pub trait AxisType {
    fn usize(&self) -> usize;
}

impl AxisType for Axis {
    fn usize(&self) -> usize {
        self.0
    }
}

impl AxisType for usize {
    fn usize(&self) -> usize {
        *self
    }
}
