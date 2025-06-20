// Trait which recursively flattens a vector
// [0, 50, 100] -> [0, 50, 100]
// [[0], [50, 50], 100] -> [0, 50, 50, 100]
// [[[0]]] -> [0]

use crate::recursive_trait_base_cases;
use crate::util::shape::Shape;
use std::fmt::Debug;

use std::ptr::copy_nonoverlapping;

pub(crate) trait Flatten<A> {
    fn flatten(self) -> Vec<A>;
}

impl<A, T> Flatten<A> for Vec<T>
where
    T: Flatten<A>,
{
    fn flatten(self) -> Vec<A> {
        self.into_iter()
            .flat_map(|nested| nested.flatten().into_iter())
            .collect()
    }
}

impl<A, T, const N: usize> Flatten<A> for [T; N]
where
    T: Flatten<A>,
    [T; N]: Shape,
    A: Debug,
{
    fn flatten(mut self) -> Vec<A> {
        assert!(align_of::<T>() >= align_of::<A>(), "alignment mismatch");  // ChatGPT suggested

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
