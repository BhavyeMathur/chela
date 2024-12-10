use crate::recursive_vec_trait_base_cases;

pub trait Shape {
    fn shape(&self) -> Vec<usize>;
}

impl<T> Shape for Vec<T>
where
    T: Shape,
{
    fn shape(&self) -> Vec<usize> {
        [vec![self.len()], self[0].shape()].concat()
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

recursive_vec_trait_base_cases!(shape_vec_trait);
