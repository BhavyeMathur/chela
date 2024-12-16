use crate::Axis;

use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use crate::axis::indexer_impl::IndexerImpl;

pub(crate) trait Indexer: IndexerImpl + Clone {
    fn indexed_shape_and_stride(&self, axis: &Axis, shape: &[usize], stride: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let mut shape = shape.to_vec();
        let mut stride = stride.to_vec();

        let axis = axis.0;
        let len = self.len(axis, &shape);

        if len == 0 {
            shape.remove(axis);
            stride.remove(axis);
        } else {
            shape[axis] = len;
        }

        (shape, stride)
    }

    fn index_of_first_element(&self) -> usize;
}

impl Indexer for usize {
    fn index_of_first_element(&self) -> usize {
        *self
    }
}
impl Indexer for Range<usize> {
    fn index_of_first_element(&self) -> usize {
        self.start
    }
}
impl Indexer for RangeFull {
    fn index_of_first_element(&self) -> usize {
        0
    }
}
impl Indexer for RangeFrom<usize> {
    fn index_of_first_element(&self) -> usize {
        self.start
    }
}
impl Indexer for RangeTo<usize> {
    fn index_of_first_element(&self) -> usize {
        0
    }
}
impl Indexer for RangeToInclusive<usize> {
    fn index_of_first_element(&self) -> usize {
        0
    }
}
impl Indexer for RangeInclusive<usize> {
    fn index_of_first_element(&self) -> usize {
        *self.start()
    }
}
