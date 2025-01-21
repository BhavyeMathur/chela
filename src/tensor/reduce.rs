use crate::dtype::{NumericDataType, RawDataType};
use crate::traits::to_vec::ToVec;
use crate::Tensor;


impl<T: RawDataType> Tensor<'_, T> {
    pub fn reduce<D: RawDataType>(&self, func: impl Fn(Tensor<T>) -> D, axes: impl ToVec<usize>) -> Tensor<D> {
        let axes = axes.to_vec();

        let mut shape_mask = vec![false; self.ndims()];
        for i in 0..axes.len() {
            shape_mask[axes[i]] = true;
        }

        let mut reduced_shape = Vec::with_capacity(self.ndims() - axes.len());

        for i in 0..self.shape.len() {
            if shape_mask[i] {
                reduced_shape.push(self.shape[i]);
            }
        }

        let mut output = Vec::with_capacity(reduced_shape.iter().product());
        for slice in self.nditer(axes) {
            output.push(func(slice));
        }

        unsafe { Tensor::from_contiguous_owned_buffer(reduced_shape, output) }
    }
}

impl<T: NumericDataType> Tensor<'_, T> {
    pub fn sum(&self) -> Tensor<T> {
        self.sum_along([])
    }

    pub fn product(&self) -> Tensor<T> {
        self.product_along([])
    }

    pub fn mean(&self) -> Tensor<T::AsFloatType> {
        self.mean_along([])
    }

    pub fn max(&self) -> Tensor<T> {
        self.max_along([])
    }

    pub fn min(&self) -> Tensor<T> {
        self.min_along([])
    }

    pub fn sum_along(&self, axes: impl ToVec<usize>) -> Tensor<T> {
        self.reduce(|x| x.flatiter().sum::<T>().into(), axes)
    }

    pub fn product_along(&self, axes: impl ToVec<usize>) -> Tensor<T> {
        self.reduce(|x| x.flatiter().product::<T>().into(), axes)
    }

    pub fn mean_along(&self, axes: impl ToVec<usize>) -> Tensor<T::AsFloatType> {
        let axes = axes.to_vec();
        let den = self.slice(axes.clone()).size().to_float().into();

        self.reduce(|x| x.flatiter().sum::<T>().to_float() / den, axes)
    }

    pub fn max_along(&self, axes: impl ToVec<usize>) -> Tensor<T> {
        self.reduce(|x| x.flatiter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(), axes)
    }

    pub fn min_along(&self, axes: impl ToVec<usize>) -> Tensor<T> {
        self.reduce(|x| x.flatiter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(), axes)
    }
}