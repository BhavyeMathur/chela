use crate::dtype::{BoundedRange, RawDataType};
use crate::traits::to_vec::ToVec;
use crate::Tensor;
use num::{Float, Num};
use rand::distributions::Distribution;
use rand::{thread_rng, Rng};
use rand_distr::Normal;
use crate::index::Index::Range;

impl<T: RawDataType + Float> Tensor<'_, T> {
    pub fn randn(shape: impl ToVec<usize>) -> Self {
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
}

impl<T: RawDataType + PartialOrd + Num + rand::distributions::uniform::SampleUniform>
    Tensor<'_, T>
{
    pub fn rand<R: BoundedRange<T>>(range: R, shape: impl ToVec<usize>) -> Self {
        let mut rng = thread_rng();
        let shape = shape.to_vec();
        let n = shape.iter().product();

        let start = *range.start_bound();
        let end = *range.end_bound();

        let mut random_numbers;

        if range.is_inclusive(){
            random_numbers = (0..n)
                .map(|_| T::from(rng.gen_range(start..=end))) // Generate normal random number and cast it to T
                .collect();
        }

        else {
            random_numbers = (0..n)
                .map(|_| T::from(rng.gen_range(start..end))) // Generate normal random number and cast it to T
                .collect();
        }

        unsafe { Tensor::from_contiguous_owned_buffer(shape, random_numbers) }
    }
}
