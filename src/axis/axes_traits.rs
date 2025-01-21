use crate::traits::haslength::HasLength;
use crate::Axis;

pub trait AxisType {
    fn usize(&self) -> usize;
}

impl AxisType for Axis {
    fn usize(&self) -> usize {
        self.0
    }
}

impl AxisType for usize {
    fn usize(&self) -> usize {
        *self
    }
}

pub trait AxesType: IntoIterator<Item=usize> + HasLength + Clone {}

impl<const N: usize> AxesType for [usize; N] {}

impl AxesType for Vec<usize> {}
