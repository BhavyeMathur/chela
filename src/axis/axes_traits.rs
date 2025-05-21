use crate::traits::haslength::HasLength;
use crate::Axis;

pub trait AxisType {
    fn isize(&self) -> isize;
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
