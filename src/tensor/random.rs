use std::ops::Range;
use num::{Float, Num, ToPrimitive};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use rand_distr::Normal;
use crate::dtype::RawDataType;
use crate::Tensor;
use crate::traits::to_vec::ToVec;

impl<T: RawDataType> Tensor<'_, T> {
    pub fn randn(shape: impl ToVec<usize>) -> Self
    where
        T: Float
    {
        let mut rng = thread_rng();
        let shape = shape.to_vec();
        let n = shape.iter().product();

        let normal = Normal::new(0.0, 1.0).unwrap(); // Standard normal distribution (mean 0, stddev 1)

        // Generate n random numbers from a standard normal distribution
        let random_numbers: Vec<T> = (0..n)
            .map(|_| T::from(normal.sample(&mut rng)).unwrap()) // Generate normal random number and cast it to T
            .collect();

        unsafe { Tensor::from_contiguous_owned_buffer(shape, random_numbers) }
    }

    pub fn rand(range: Range<T>, shape: impl ToVec<usize>) -> Self
    where
        T: Num + rand::distributions::uniform::SampleUniform + ToPrimitive
    {
        let mut rng = thread_rng();
        let shape = shape.to_vec();
        let n = shape.iter().product();

        let uniform = Uniform::new_inclusive(range.start, range.end);

        // Generate n random numbers from a standard normal distribution
        let random_numbers: Vec<T> = (0..n)
            .map(|_| T::from(uniform.sample(&mut rng))) // Generate normal random number and cast it to T
            .collect();

        unsafe { Tensor::from_contiguous_owned_buffer(shape, random_numbers) }
    }
}