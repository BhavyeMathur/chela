use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

#[macro_export]
macro_rules! s {
    ($($x:expr),*) => {
        [$($crate::util::index::Index::from($x)),*]
    };
}

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

pub(crate) trait Indexer: Clone {
    /// The resulting dimension of the axis indexed by this indexer.
    fn indexed_length(&self, axis_length: usize) -> usize;
    
    /// The first element along the dimension indexed by this kind of indexer
    /// For example, 0 for `tensor[..]` or `tensor[..2]` but 5 for `tensor[5..]` or `tensor[5]`
    fn index_of_first_element(&self) -> usize;

    /// When indexed with this kind of object, does the dimension of the tensor collapse?
    /// Only true for usize since all range-based indexers retain the dimension.
    fn collapse_dimension(&self) -> bool {
        false
    }
}

impl Indexer for Index {
    fn indexed_length(&self, len: usize) -> usize {
        match self {
            Index::Usize(index) => Indexer::indexed_length(index, len),
            Index::Range(index) => Indexer::indexed_length(index, len),
            Index::RangeFrom(index) => Indexer::indexed_length(index, len),
            Index::RangeFull(index) => Indexer::indexed_length(index, len),
            Index::RangeInclusive(index) => Indexer::indexed_length(index, len),
            Index::RangeTo(index) => Indexer::indexed_length(index, len),
            Index::RangeToInclusive(index) => Indexer::indexed_length(index, len),
        }
    }

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

impl Indexer for usize {
    fn indexed_length(&self, _axis_length: usize) -> usize {
        1
    }

    fn index_of_first_element(&self) -> usize {
        *self
    }

    fn collapse_dimension(&self) -> bool {
        true
    }
}
impl Indexer for Range<usize> {
    fn indexed_length(&self, _axis_length: usize) -> usize {
        self.end - self.start
    }

    fn index_of_first_element(&self) -> usize {
        self.start
    }
}
impl Indexer for RangeFull {
    fn indexed_length(&self, axis_length: usize) -> usize {
        axis_length
    }
    fn index_of_first_element(&self) -> usize {
        0
    }
}
impl Indexer for RangeFrom<usize> {
    fn indexed_length(&self, axis_length: usize) -> usize {
        axis_length - self.start
    }
    fn index_of_first_element(&self) -> usize {
        self.start
    }
}
impl Indexer for RangeTo<usize> {
    fn indexed_length(&self, _axis_length: usize) -> usize {
        self.end
    }
    fn index_of_first_element(&self) -> usize {
        0
    }
}
impl Indexer for RangeToInclusive<usize> {
    fn indexed_length(&self, _axis_length: usize) -> usize {
        self.end + 1
    }
    fn index_of_first_element(&self) -> usize {
        0
    }
}
impl Indexer for RangeInclusive<usize> {
    fn indexed_length(&self, _axis_length: usize) -> usize {
        self.end() - self.start() + 1
    }
    fn index_of_first_element(&self) -> usize {
        *self.start()
    }
}

impl From<usize> for Index {
    fn from(val: usize) -> Self {
        Index::Usize(val)
    }
}

impl From<Range<usize>> for Index {
    fn from(val: Range<usize>) -> Self {
        Index::Range(val)
    }
}

impl From<RangeFrom<usize>> for Index {
    fn from(val: RangeFrom<usize>) -> Self {
        Index::RangeFrom(val)
    }
}

impl From<RangeFull> for Index {
    fn from(val: RangeFull) -> Self {
        Index::RangeFull(val)
    }
}

impl From<RangeInclusive<usize>> for Index {
    fn from(val: RangeInclusive<usize>) -> Self {
        Index::RangeInclusive(val)
    }
}

impl From<RangeTo<usize>> for Index {
    fn from(val: RangeTo<usize>) -> Self {
        Index::RangeTo(val)
    }
}

impl From<RangeToInclusive<usize>> for Index {
    fn from(val: RangeToInclusive<usize>) -> Self {
        Index::RangeToInclusive(val)
    }
}
