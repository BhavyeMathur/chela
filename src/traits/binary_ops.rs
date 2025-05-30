use crate::StridedMemory;
use crate::broadcast::broadcast_shapes;
use crate::{IntegerDataType, NdArray, RawDataType};
use std::ops::{Add, BitAnd, BitOr, Div, Mul, Rem, Shl, Shr, Sub};

use paste::paste;

macro_rules! define_binary_ops {
    ($object:ident, $($trait_: ident, $operator: tt, $method: ident;)* ) => {
        $(
            fn $method<'a, 'b>(lhs: impl AsRef<$object<'a, T>>,
                               rhs: impl AsRef<$object<'b, T>>) -> $object<'static, T>
            where
                T: $trait_<Output=T>,
            {
                let lhs = lhs.as_ref();
                let rhs = rhs.as_ref();

                let shape = broadcast_shapes(lhs.shape(), rhs.shape());
                let lhs = lhs.broadcast_to(&shape);
                let rhs = rhs.broadcast_to(&shape);

                let data = lhs.flatiter().zip(rhs.flatiter()).map(|(lhs, rhs)| lhs $operator rhs).collect();
                unsafe { $object::from_contiguous_owned_buffer(shape, data, false, false) }
            }

            paste! { fn [<$method _scalar>] <'a, 'b>(lhs: impl AsRef<$object<'a, T>>,
                                                     rhs: T) -> $object<'static, T>
                where
                    T: $trait_<Output=T>,
                {
                    let lhs = lhs.as_ref();

                    let data = lhs.flatiter().map(|lhs| lhs $operator rhs).collect();
                    unsafe { $object::from_contiguous_owned_buffer(lhs.shape().to_vec(), data, false, false) }
                }
            }
        )*
    }
}

pub(crate) trait BinaryOps<T: RawDataType> {
    define_binary_ops!(
        NdArray,
        Add, +, add;
        Sub, -, sub;
        Mul, *, mul;
        Div, /, div;
        Rem, %, rem;
        BitAnd, &, bitand;
        BitOr, |, bitor;
        Shl, <<, shl;
        Shr, >>, shr;
    );
}

impl<T: IntegerDataType> BinaryOps<T> for T {}
impl BinaryOps<bool> for bool {}
impl BinaryOps<f32> for f32 {}
impl BinaryOps<f64> for f64 {}
