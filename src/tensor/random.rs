use crate::dtype::RawDataType;
use crate::traits::to_vec::ToVec;
use crate::Tensor;
use num::Float;
use rand::distributions::{Distribution, Uniform, uniform::SampleUniform};
use rand::thread_rng;
use rand_distr::Normal;

impl<T: RawDataType + Float> Tensor<'_, T> {
    pub fn randn(shape: impl ToVec<usize>) -> Self {
        let mut rng = thread_rng();
        let shape = shape.to_vec();
        let n = shape.iter().product();

        let normal = Normal::new(0.0, 1.0).unwrap();

        let random_numbers: Vec<T> = (0..n)
            .map(|_| T::from(normal.sample(&mut rng)).unwrap())
            .collect();

        unsafe { Tensor::from_contiguous_owned_buffer(shape, random_numbers) }
    }
}

impl<T: RawDataType + SampleUniform> Tensor<'_, T> {
    pub fn rand(shape: impl ToVec<usize>) -> Self {
        let mut rng = thread_rng();
        let shape = shape.to_vec();
        let n = shape.iter().product();

        let uniform = Uniform::new(0.0, 1.0);
        let random_numbers = (0..n)
            .map(|_| T::from(uniform.sample(&mut rng)).unwrap())
            .collect();

        unsafe { Tensor::from_contiguous_owned_buffer(shape, random_numbers) }
    }
}
