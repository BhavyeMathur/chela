use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

pub(crate) trait IndexerImpl {
    fn indexed_length(&self, axis_length: usize) -> usize;
}

impl IndexerImpl for usize {
    fn indexed_length(&self, _axis_length: usize) -> usize {
        1
    }
}
impl IndexerImpl for Range<usize> {
    fn indexed_length(&self, _axis_length: usize) -> usize {
        self.end - self.start
    }
}

impl IndexerImpl for RangeFull {
    fn indexed_length(&self, axis_length: usize) -> usize {
        axis_length
    }
}

impl IndexerImpl for RangeFrom<usize> {
    fn indexed_length(&self, axis_length: usize) -> usize {
        axis_length - self.start
    }
}

impl IndexerImpl for RangeTo<usize> {
    fn indexed_length(&self, _axis_length: usize) -> usize {
        self.end
    }
}

impl IndexerImpl for RangeInclusive<usize> {
    fn indexed_length(&self, _axis_length: usize) -> usize {
        self.end() - self.start() + 1
    }
}

impl IndexerImpl for RangeToInclusive<usize> {
    fn indexed_length(&self, _axis_length: usize) -> usize {
        self.end + 1
    }
}
