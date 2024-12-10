// Trait which recursively flattens a vector
// [0, 50, 100] -> [0, 50, 100]
// [[0], [50, 50], 100] -> [0, 50, 50, 100]
// [[[0]]] -> [0]

use crate::tensor::dtype::RawDataType;
use crate::recursive_trait_base_cases;

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

macro_rules! flatten_vec_trait {
    ( $dtype:ty ) => {
        impl Flatten<$dtype> for Vec<$dtype> {
            fn flatten(self) -> Vec<$dtype> {
                self
            }
        }
    };
}

recursive_trait_base_cases!(flatten_vec_trait);
