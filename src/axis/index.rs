use crate::axis::indexer::Indexer;
use crate::axis::indexer_impl::IndexerImpl;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

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

    fn collapse_dimension(&self) -> bool {
        matches!(self, Index::Usize(_))
    }
}

impl IndexerImpl for Index {
    fn indexed_length(&self, len: usize) -> usize {
        match self {
            Index::Usize(index) => IndexerImpl::indexed_length(index, len),
            Index::Range(index) => IndexerImpl::indexed_length(index, len),
            Index::RangeFrom(index) => IndexerImpl::indexed_length(index, len),
            Index::RangeFull(index) => IndexerImpl::indexed_length(index, len),
            Index::RangeInclusive(index) => IndexerImpl::indexed_length(index, len),
            Index::RangeTo(index) => IndexerImpl::indexed_length(index, len),
            Index::RangeToInclusive(index) => IndexerImpl::indexed_length(index, len),
        }
    }
}
