// Trait which recursively flattens a vector
// [0, 50, 100] -> [0, 50, 100]
// [[0], [50, 50], 100] -> [0, 50, 50, 100]
// [[[0]]] -> [0]

use crate::recursive_trait_base_cases;
use crate::tensor::dtype::RawDataType;
use crate::traits::shape::Shape;
use std::fmt::Debug;

use std::ptr::copy_nonoverlapping;

pub trait Flatten<A: RawDataType> {
    fn flatten(self) -> Vec<A>;
}

impl<A: RawDataType, T> Flatten<A> for Vec<T>
where
    T: Flatten<A>,
{
    fn flatten(self) -> Vec<A> {
        self.into_iter()
            .flat_map(|nested| nested.flatten().into_iter())
            .collect()

        // TODO deed to test performance with the following
        // let len = self.shape().iter().product();
        // let mut result = Vec::with_capacity(len);
        //
        // for nested in self {
        //     result.append(nested.flatten().as_mut());
        // }
        // result
    }
}

impl<A: RawDataType, T, const N: usize> Flatten<A> for [T; N]
where
    T: Flatten<A>,
    [T; N]: Shape,
    A: Debug,
{
    fn flatten(mut self) -> Vec<A> {
        // ChatGPT suggested
        assert!(align_of::<T>() >= align_of::<A>(), "alignment mismatch");

        let len = self.shape().iter().product();
        let mut result = Vec::with_capacity(len);

        let src = self.as_mut_ptr() as *mut A;
        let dst = result.as_mut_ptr();

        unsafe {
            copy_nonoverlapping(src, dst, len);
            result.set_len(len);
        }
        result
    }
}

macro_rules! flatten_trait {
    ( $dtype:ty ) => {
        impl Flatten<$dtype> for Vec<$dtype> {
            fn flatten(self) -> Vec<$dtype> {
                self
            }
        }

        impl<const N: usize> Flatten<$dtype> for [$dtype; N] {
            fn flatten(self) -> Vec<$dtype> {
                Vec::from(self)
            }
        }
    };
}

recursive_trait_base_cases!(flatten_trait);
