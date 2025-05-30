use crate::util::to_vec::ToVec;
use crate::{FloatDataType, NumericDataType, NdArray};
use num::NumCast;
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use rand_distr::Normal;

impl<T: FloatDataType + SampleUniform> NdArray<'_, T> {
    pub fn randn(shape: impl ToVec<usize>) -> Self {
        let mut rng = thread_rng();
        let shape = shape.to_vec();
        let n = shape.iter().product();

        let normal = Normal::new(0.0, 1.0).unwrap();

        let random_numbers: Vec<T> = (0..n)
            .map(|_| <T as NumCast>::from(normal.sample(&mut rng)).unwrap())
            .collect();

        unsafe { NdArray::from_contiguous_owned_buffer(shape, random_numbers) }
    }

    pub fn rand(shape: impl ToVec<usize>) -> Self {
        let mut rng = thread_rng();
        let shape = shape.to_vec();
        let n = shape.iter().product();

        let uniform = Uniform::new(0.0, 1.0);
        let random_numbers = (0..n)
            .map(|_| <T as NumCast>::from(uniform.sample(&mut rng)).unwrap())
            .collect();

        unsafe { NdArray::from_contiguous_owned_buffer(shape, random_numbers) }
    }
}

impl<T: NumericDataType> NdArray<'_, T> {
    pub fn randint(shape: impl ToVec<usize>, low: T, high: T) -> Self {
        assert!(low < high, "randint: low must be less than high");

        let mut rng = thread_rng();
        let shape = shape.to_vec();
        let n = shape.iter().product();

        let uniform = Uniform::new(low.to_float(), high.to_float());
        let random_numbers = (0..n)
            .map(|_| <T as NumCast>::from(uniform.sample(&mut rng)).unwrap())
            .collect();

        unsafe { NdArray::from_contiguous_owned_buffer(shape, random_numbers) }
    }
}
