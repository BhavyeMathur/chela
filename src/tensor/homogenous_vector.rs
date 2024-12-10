// Trait to check if a vector has homogeneously sized (rectangular) dimensions
// [[0], [1], [2]] - homogenous
// [[0, 1], [2]] - not homogenous

use crate::recursive_trait_base_cases;
use super::shape::Shape;

pub trait HomogenousVec: Shape {
    fn check_homogenous(&self) -> bool;
}

impl<T> HomogenousVec for Vec<T>
where
    T: HomogenousVec,
{
    fn check_homogenous(&self) -> bool {
        let first_length = self[0].shape();

        self.iter()
            .all(|v| v.check_homogenous() && v.shape() == first_length)
    }
}

macro_rules! homogenous_vec_trait {
    ( $dtype: ty ) => {
        impl HomogenousVec for Vec<$dtype> {
            fn check_homogenous(&self) -> bool {
                true
            }
        }
    };
}

recursive_trait_base_cases!(homogenous_vec_trait);
