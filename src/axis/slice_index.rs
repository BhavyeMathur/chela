use crate::axis::index::Index;

use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

#[macro_export]
macro_rules! s {
    ($($x:expr),*) => {
        [$($crate::axis::index::Index::from($x)),*]
    };
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
