use crate::broadcast::{broadcast_shapes, broadcast_stride};
use crate::common::constructors::Constructors;
use crate::StridedMemory;
use crate::{IntegerDataType, NdArray, RawDataType};
use std::ops::{Add, BitAnd, BitOr, Div, Mul, Rem, Shl, Shr, Sub};

use crate::ops::binary_op_addition::BinaryOpAdd;
use paste::paste;

macro_rules! define_binary_ops {
    ($object:ident, $($trait_: ident, $operator: tt, $method: ident;)* ) => {
        $(
            paste! {
                fn $method<'a, 'b>(lhs: impl AsRef<$object<'a, Self>>,
                               rhs: impl AsRef<$object<'b, Self>>) -> $object<'static, Self>
                where
                    Self: $trait_<Output=Self> + RawDataType,
                {
                    let lhs = lhs.as_ref();
                    let rhs = rhs.as_ref();

                    let shape = broadcast_shapes(lhs.shape(), rhs.shape());
                    let lhs = lhs.broadcast_to(&shape);
                    let rhs = rhs.broadcast_to(&shape);

                    let data = lhs.flatiter().zip(rhs.flatiter()).map(|(lhs, rhs)| lhs $operator rhs).collect();
                    unsafe { $object::from_contiguous_owned_buffer(shape, data) }
                }
            }

            paste! { fn [<$method _scalar>] <'a, 'b>(lhs: impl AsRef<$object<'a, Self>>,
                                                     rhs: Self) -> $object<'static, Self>
                where
                    Self: $trait_<Output=Self> + RawDataType,
                {
                    let lhs = lhs.as_ref();

                    let data = lhs.flatiter().map(|lhs| lhs $operator rhs).collect();
                    unsafe { $object::from_contiguous_owned_buffer(lhs.shape().to_vec(), data) }
                }
            }
        )*
    }
}

pub(crate) trait BinaryOps: Sized + Copy {
    fn add<'a, 'b>(lhs: impl AsRef<NdArray<'a, Self>>,
                   rhs: impl AsRef<NdArray<'b, Self>>) -> NdArray<'static, Self>
    where
        Self: RawDataType + BinaryOpAdd,
    {
        let lhs = lhs.as_ref();
        let rhs = rhs.as_ref();

        let shape = broadcast_shapes(lhs.shape(), rhs.shape());
        let lhs_stride = broadcast_stride(lhs.stride(), &shape, lhs.shape());
        let rhs_stride = broadcast_stride(rhs.stride(), &shape, rhs.shape());

        let mut data = vec![Self::default(); shape.iter().product()];

        unsafe {
            <Self as BinaryOpAdd>::add(lhs.ptr(), &lhs_stride,
                                       rhs.ptr(), &rhs_stride,
                                       data.as_mut_ptr(), &shape);
        }

        unsafe { NdArray::from_contiguous_owned_buffer(shape, data) }
    }

    fn add_scalar<'a, 'b>(lhs: impl AsRef<NdArray<'a, Self>>,
                          rhs: Self) -> NdArray<'static, Self>
    where
        Self: Add<Output=Self> + RawDataType,
    {
        let lhs = lhs.as_ref();

        let data = lhs.flatiter().map(|lhs| lhs + rhs).collect();
        unsafe { NdArray::from_contiguous_owned_buffer(lhs.shape().to_vec(), data) }
    }

    define_binary_ops!(
        NdArray,
        // Add, +, add;
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

impl<T: IntegerDataType> BinaryOps for T {}
impl BinaryOps for bool {}
impl BinaryOps for f32 {}
impl BinaryOps for f64 {}
