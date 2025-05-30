use crate::util::to_vec::ToVec;
use crate::{FloatDataType, NdArray, NumericDataType};
use num::{Float, NumCast};
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use rand_distr::Normal;

impl<T: FloatDataType + SampleUniform> NdArray<'_, T> {
    /// Samples an `NdArray` with the specified shape
    /// from a standard normal distribution (0 mean, unit standard deviation).
    ///
    /// # Examples
    /// ```
    /// # use chela::*;
    ///
    /// let ndarray = NdArray::<f64>::randn([2, 3]);
    /// println!("{:?}", ndarray);
    /// ```
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

    /// Samples an `NdArray` with the specified shape
    /// with values uniformly distributed in [0, 1).
    ///
    /// # Examples
    /// ```
    /// # use chela::*;
    ///
    /// let ndarray = NdArray::<f64>::rand([2, 3]);
    /// println!("{:?}", ndarray);
    /// ```
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

    /// Samples an `NdArray` with the specified shape
    /// with values uniformly distributed in [`low`, `high`).
    ///
    /// # Examples
    /// ```
    /// # use chela::*;
    ///
    /// let ndarray = NdArray::<f64>::uniform([2, 3], -5.0, 3.0);
    /// println!("{:?}", ndarray);
    /// ```
    pub fn uniform(shape: impl ToVec<usize>, low: T, high: T) -> Self {
        let mut rng = thread_rng();
        let shape = shape.to_vec();
        let n = shape.iter().product();

        let uniform = Uniform::new(low, high);
        let random_numbers = (0..n)
            .map(|_| <T as NumCast>::from(uniform.sample(&mut rng)).unwrap())
            .collect();

        unsafe { NdArray::from_contiguous_owned_buffer(shape, random_numbers) }
    }
}

impl<T: NumericDataType> NdArray<'_, T> {
    /// Samples an `NdArray` with the specified shape
    /// with integer values uniformly distributed between `low` (inclusive) and `high` (exclusive).
    ///
    /// # Examples
    /// ```
    /// # use chela::*;
    ///
    /// let ndarray = NdArray::<isize>::randint([2, 3], -5, 3);
    /// println!("{:?}", ndarray);
    /// ```
    pub fn randint(shape: impl ToVec<usize>, low: T, high: T) -> Self {
        assert!(low < high, "randint: low must be less than high");

        let mut rng = thread_rng();
        let shape = shape.to_vec();
        let n = shape.iter().product();

        let uniform = Uniform::new(low.to_float(), high.to_float());
        let random_numbers = (0..n)
            .map(|_| <T as NumCast>::from(uniform.sample(&mut rng).round()).unwrap())
            .collect();

        unsafe { NdArray::from_contiguous_owned_buffer(shape, random_numbers) }
    }
}
