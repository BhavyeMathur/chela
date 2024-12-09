use crate::tensor::dtype::RawDataType;

pub trait FlattenVec<A> {
    fn flatten(self) -> Vec<A>;
}

impl<A, T> FlattenVec<A> for Vec<T>
where
    T: FlattenVec<A>,
    A: RawDataType
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

flatten_vec_trait!(i8);
flatten_vec_trait!(i16);
flatten_vec_trait!(i32);
flatten_vec_trait!(i64);
flatten_vec_trait!(i128);

flatten_vec_trait!(u8);
flatten_vec_trait!(u16);
flatten_vec_trait!(u32);
flatten_vec_trait!(u64);
flatten_vec_trait!(u128);

flatten_vec_trait!(f32);
flatten_vec_trait!(f64);

flatten_vec_trait!(bool);
