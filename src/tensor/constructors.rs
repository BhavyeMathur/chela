use crate::tensor::data_owned::DataOwned;
use crate::tensor::dtype::RawDataType;
use crate::tensor::Tensor;

use crate::traits::flatten::Flatten;
use crate::traits::homogenous::Homogenous;
use crate::traits::nested::Nested;
use crate::traits::shape::Shape;

impl<T: RawDataType, const D: usize> Tensor<T, D> {
    pub fn from(data: impl Flatten<T> + Homogenous + Shape + Nested<D>) -> Self {
        assert!(
            data.check_homogenous(),
            "Tensor::from() failed, found inhomogeneous dimensions"
        );

        let shape = data.shape().try_into().unwrap();
        let data = DataOwned::from(data);

        Self { data, shape }
    }
}
