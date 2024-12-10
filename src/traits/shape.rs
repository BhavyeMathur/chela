use crate::recursive_trait_base_cases;

pub trait Shape {
    fn shape(&self) -> Vec<usize>;
}

impl<T: Shape> Shape for Vec<T> {
    fn shape(&self) -> Vec<usize> {
        [vec![self.len()], self[0].shape()].concat()
    }
}

impl<T: Shape, const N: usize> Shape for [T; N] {
    fn shape(&self) -> Vec<usize> {
        [vec![self.len()], self[0].shape()].concat()
    }
}

macro_rules! shape_trait {
    ( $dtype: ty ) => {
        impl Shape for Vec<$dtype> {
            fn shape(&self) -> Vec<usize> {
                vec![self.len()]
            }
        }

        impl<const N: usize> Shape for [$dtype; N] {
            fn shape(&self) -> Vec<usize> {
                vec![self.len()]
            }
        }
    };
}

recursive_trait_base_cases!(shape_trait);
