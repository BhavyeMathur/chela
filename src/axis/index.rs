use crate::axis::indexer::Indexer;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use crate::axis::indexer_impl::IndexerImpl;

#[derive(Clone)]
pub enum Index {
    Usize(usize),
    Range(Range<usize>),
    RangeFrom(RangeFrom<usize>),
    RangeFull(RangeFull),
    RangeInclusive(RangeInclusive<usize>),
    RangeTo(RangeTo<usize>),
    RangeToInclusive(RangeToInclusive<usize>),
}

impl Indexer for Index {
    fn index_of_first_element(&self) -> usize {
        match self {
            Index::Usize(index) => index.index_of_first_element(),
            Index::Range(index) => index.index_of_first_element(),
            Index::RangeFrom(index) => index.index_of_first_element(),
            Index::RangeFull(index) => index.index_of_first_element(),
            Index::RangeInclusive(index) => index.index_of_first_element(),
            Index::RangeTo(index) => index.index_of_first_element(),
            Index::RangeToInclusive(index) => index.index_of_first_element(),
        }
    }
}

impl IndexerImpl for Index {
    fn len(&self, axis: usize, shape: &[usize]) -> usize {
        match self {
            Index::Usize(index) => IndexerImpl::len(index, axis, shape),
            Index::Range(index) => IndexerImpl::len(index, axis, shape),
            Index::RangeFrom(index) => IndexerImpl::len(index, axis, shape),
            Index::RangeFull(index) => IndexerImpl::len(index, axis, shape),
            Index::RangeInclusive(index) => IndexerImpl::len(index, axis, shape),
            Index::RangeTo(index) => IndexerImpl::len(index, axis, shape),
            Index::RangeToInclusive(index) => IndexerImpl::len(index, axis, shape),
        }
    }
}
