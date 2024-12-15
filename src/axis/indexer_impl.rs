use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

pub(super) trait IndexerImpl {
    fn len(&self, axis: usize, shape: &[usize]) -> usize;
}

impl IndexerImpl for usize {
    fn len(&self, _axis: usize, _shape: &[usize]) -> usize {
        0
    }
}
impl IndexerImpl for Range<usize> {
    fn len(&self, _axis: usize, _shape: &[usize]) -> usize {
        self.end - self.start
    }
}

impl IndexerImpl for RangeFull {
    fn len(&self, axis: usize, shape: &[usize]) -> usize {
        shape[axis]
    }
}

impl IndexerImpl for RangeFrom<usize> {
    fn len(&self, axis: usize, shape: &[usize]) -> usize {
        shape[axis] - self.start
    }
}

impl IndexerImpl for RangeTo<usize> {
    fn len(&self, _axis: usize, _shape: &[usize]) -> usize {
        self.end
    }
}

impl IndexerImpl for RangeInclusive<usize> {
    fn len(&self, _axis: usize, _shape: &[usize]) -> usize {
        self.end() - self.start() + 1
    }
}

impl IndexerImpl for RangeToInclusive<usize> {
    fn len(&self, _axis: usize, _shape: &[usize]) -> usize {
        self.end + 1
    }
}
