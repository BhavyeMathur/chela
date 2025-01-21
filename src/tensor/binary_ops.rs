use crate::broadcast::broadcast_shapes;
use crate::dtype::RawDataType;
use crate::Tensor;
use std::ops::Add;

impl<T: RawDataType + Add<Output=T> + 'static> Add for Tensor<'_, T> {
    type Output = Tensor<'static, T>;

    fn add(self, rhs: Tensor<T>) -> Self::Output {
        &self + rhs
    }
}

impl<T: RawDataType + Add<Output=T> + 'static> Add<&Tensor<'_, T>> for Tensor<'_, T> {
    type Output = Tensor<'static, T>;

    fn add(self, rhs: &Tensor<T>) -> Self::Output {
        &self + rhs
    }
}

impl<T: RawDataType + Add<Output=T> + 'static> Add<Tensor<'_, T>> for &Tensor<'_, T> {
    type Output = Tensor<'static, T>;

    fn add(self, rhs: Tensor<T>) -> Self::Output {
        self + &rhs
    }
}

impl<T: RawDataType + Add<Output=T> + 'static> Add<&Tensor<'_, T>> for &Tensor<'_, T> {
    type Output = Tensor<'static, T>;

    fn add(self, rhs: &Tensor<T>) -> Self::Output {
        let shape = broadcast_shapes(&self.shape, &rhs.shape);
        let lhs = self.broadcast_to(&shape);
        let rhs = rhs.broadcast_to(&shape);

        let data = lhs.flatiter().zip(rhs.flatiter()).map(|(lhs, rhs)| lhs + rhs).collect();
        unsafe { Tensor::from_contiguous_owned_buffer(shape, data) }
    }
}

impl<T: RawDataType + Add<Output=T> + 'static> Add<T> for Tensor<'_, T> {
    type Output = Tensor<'static, T>;

    fn add(self, rhs: T) -> Self::Output {
        &self + rhs
    }
}

impl<T: RawDataType + Add<Output=T> + 'static> Add<T> for &Tensor<'_, T> {
    type Output = Tensor<'static, T>;

    fn add(self, rhs: T) -> Self::Output {
        let data = self.flatiter().map(|lhs| lhs + rhs).collect();
        unsafe { Tensor::from_contiguous_owned_buffer(self.shape.clone(), data) }
    }
}
