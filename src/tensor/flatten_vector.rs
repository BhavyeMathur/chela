// Trait which recursively flattens a vector
// [0, 50, 100] -> [0, 50, 100]
// [[0], [50, 50], 100] -> [0, 50, 50, 100]
// [[[0]]] -> [0]

use super::dtype::RawDataType;
use crate::recursive_vec_trait_base_cases;

pub trait FlattenVec<A> {
    fn flatten(self) -> Vec<A>;
}

impl<A, T> FlattenVec<A> for Vec<T>
where
    T: FlattenVec<A>,
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
        impl FlattenVec<$dtype> for Vec<$dtype> {
            fn flatten(self) -> Vec<$dtype> {
                self
            }
        }
    };
}

recursive_vec_trait_base_cases!(flatten_vec_trait);
