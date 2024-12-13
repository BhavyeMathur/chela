use crate::Axis;

use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

pub(crate) trait Index: IndexImpl + Clone {
    fn indexed_shape_and_stride(&self, axis: &Axis, shape: &Vec<usize>, stride: &Vec<usize>) -> (Vec<usize>, Vec<usize>) {
        let mut shape = shape.clone();
        let mut stride = stride.clone();

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

impl Index for usize {
    fn index_of_first_element(&self) -> usize {
        *self
    }
}
impl Index for Range<usize> {
    fn index_of_first_element(&self) -> usize {
        self.start
    }
}
impl Index for RangeFull {
    fn index_of_first_element(&self) -> usize {
        0
    }
}
impl Index for RangeFrom<usize> {
    fn index_of_first_element(&self) -> usize {
        self.start
    }
}
impl Index for RangeTo<usize> {
    fn index_of_first_element(&self) -> usize {
        0
    }
}
impl Index for RangeToInclusive<usize> {
    fn index_of_first_element(&self) -> usize {
        0
    }
}
impl Index for RangeInclusive<usize> {
    fn index_of_first_element(&self) -> usize {
        *self.start()
    }
}

trait IndexImpl {
    fn len(&self, axis: usize, shape: &Vec<usize>) -> usize;
}

impl IndexImpl for usize {
    fn len(&self, _axis: usize, _shape: &Vec<usize>) -> usize {
        0
    }
}
impl IndexImpl for Range<usize> {
    fn len(&self, _axis: usize, _shape: &Vec<usize>) -> usize {
        self.end - self.start
    }
}

impl IndexImpl for RangeFull {
    fn len(&self, axis: usize, shape: &Vec<usize>) -> usize {
        shape[axis]
    }
}

impl IndexImpl for RangeFrom<usize> {
    fn len(&self, axis: usize, shape: &Vec<usize>) -> usize {
        shape[axis] - self.start
    }
}

impl IndexImpl for RangeTo<usize> {
    fn len(&self, _axis: usize, _shape: &Vec<usize>) -> usize {
        self.end
    }
}

impl IndexImpl for RangeInclusive<usize> {
    fn len(&self, _axis: usize, _shape: &Vec<usize>) -> usize {
        self.end() - self.start() + 1
    }
}

impl IndexImpl for RangeToInclusive<usize> {
    fn len(&self, _axis: usize, _shape: &Vec<usize>) -> usize {
        self.end + 1
    }
}
