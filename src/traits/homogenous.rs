// Trait to check if a vector has homogeneously sized (rectangular) dimensions
// [[0], [1], [2]] - homogenous
// [[0, 1], [2]] - not homogenous

use crate::recursive_trait_base_cases;
use crate::traits::shape::Shape;

pub trait Homogenous {
    fn check_homogenous(&self) -> bool;
}

impl<T> Homogenous for Vec<T>
where
    T: Homogenous + Shape,
{
    fn check_homogenous(&self) -> bool {
        let first_shape = self[0].shape();

        self.iter().all(|v| v.shape() == first_shape)
    }
}

impl<T, const N: usize> Homogenous for [T; N] {
    fn check_homogenous(&self) -> bool {
        true
    }
}

macro_rules! homogenous_vec_trait {
    ( $dtype: ty ) => {
        impl Homogenous for Vec<$dtype> {
            fn check_homogenous(&self) -> bool {
                true
            }
        }
    };
}

recursive_trait_base_cases!(homogenous_vec_trait);
