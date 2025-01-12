use crate::axis::indexer_impl::IndexerImpl;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

pub(crate) trait Indexer: IndexerImpl + Clone {
    fn index_of_first_element(&self) -> usize;

    fn collapse_dimension(&self) -> bool {
        false
    }
}

impl Indexer for usize {
    fn index_of_first_element(&self) -> usize {
        *self
    }

    fn collapse_dimension(&self) -> bool {
        true
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
