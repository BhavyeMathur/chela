use crate::traits::haslength::HasLength;
use crate::Axis;

pub trait AxisType {
    fn isize(&self) -> isize;

    /// Computes the absolute axis index for a given tensor dimension.
    /// 
    /// Negative axis values are normalized to represent their positive counterparts.
    /// For example, `-1` represents the last axis, `-2` the second-to-last axis, and so on.
    ///
    /// # Arguments
    ///
    /// * `ndims` - The total number of dimensions in the tensor.
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// * The provided axis is less than `-ndims` (lower bound).
    /// * The provided axis is greater than or equal to `ndims` (upper bound).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use chela::*;
    /// assert_eq!(Axis(-1).get_absolute(4), 3);
    /// assert_eq!(Axis(-2).get_absolute(4), 2);
    /// assert_eq!(Axis(1).get_absolute(4), 1);
    /// ```
    fn get_absolute(&self, ndims: usize) -> usize {
        let axis = self.isize();
        let ndims = ndims as isize;

        if axis < -ndims || axis >= ndims {
            panic!("axis '{}' out of bounds for tensor of dimension {}", axis, ndims);
        }

        (if axis < 0 { axis + ndims } else { axis }) as usize
    }
}

impl AxisType for Axis {
    fn isize(&self) -> isize {
        self.0
    }
}

impl AxisType for isize {
    fn isize(&self) -> isize {
        *self
    }
}


pub trait AxesType: IntoIterator<Item=isize> + HasLength + Clone {}

impl<const N: usize> AxesType for [isize; N] {}

impl AxesType for Vec<isize> {}
