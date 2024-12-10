use crate::recursive_trait_base_cases;

pub trait Shape {
    fn shape(&self) -> Vec<usize>;
}

impl<T> Shape for Vec<T>
where
    T: Shape,
{
    // TODO optimise this code. Right now, we recursively find the shape of each object
    // even though this can be memoized
    fn shape(&self) -> Vec<usize> {
        [vec![self.len()], self[0].shape().to_vec()].concat()
    }
}

impl<T, const N: usize> Shape for [T; N]
where
    T: Shape,
{
    fn shape(&self) -> Vec<usize> {
        [vec![self.len()], self[0].shape().to_vec()].concat()
    }
}

macro_rules! shape_vec_trait {
    ( $dtype: ty ) => {
        impl Shape for Vec<$dtype> {
            fn shape(&self) -> Vec<usize> {
                vec![self.len()]
            }
        }
    };
}

macro_rules! shape_arr_trait {
    ( $dtype: ty ) => {
        impl<const N: usize> Shape for [$dtype; N] {
            fn shape(&self) -> Vec<usize> {
                vec![self.len()]
            }
        }
    };
}

recursive_trait_base_cases!(shape_vec_trait);
recursive_trait_base_cases!(shape_arr_trait);
