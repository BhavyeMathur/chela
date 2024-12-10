// Trait which recursively flattens a vector
// [0, 50, 100] -> [0, 50, 100]
// [[0], [50, 50], 100] -> [0, 50, 50, 100]
// [[[0]]] -> [0]

use crate::recursive_trait_base_cases;
use crate::tensor::dtype::RawDataType;

pub trait Flatten<A> {
    fn flatten(self) -> Vec<A>;
}

impl<A, T> Flatten<A> for Vec<T>
where
    T: Flatten<A>,
    A: RawDataType,
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
    A: RawDataType,
{
    fn flatten(self) -> Vec<A> {
        self.into_iter()
            .flat_map(|nested| nested.flatten().into_iter())
            .collect()
    }
}

macro_rules! flatten_vec_trait {
    ( $dtype:ty ) => {
        impl Flatten<$dtype> for Vec<$dtype> {
            fn flatten(self) -> Vec<$dtype> {
                self
            }
        }
    };
}

macro_rules! flatten_array_trait {
    ( $dtype:ty ) => {
        impl<const N: usize> Flatten<$dtype> for [$dtype; N] {
            fn flatten(self) -> Vec<$dtype> {
                Vec::from(self)
            }
        }
    };
}

recursive_trait_base_cases!(flatten_vec_trait);
recursive_trait_base_cases!(flatten_array_trait);
