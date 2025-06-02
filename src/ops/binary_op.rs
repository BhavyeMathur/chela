use std::ops::{BitAnd, BitOr, Div, Rem, Shl, Shr, Sub};

#[macro_export]
macro_rules! impl_default_binary_op_trait {
    ($trait_name:ident, $($default_dtypes:ty),*) => {
        $(
            impl $trait_name for $default_dtypes {}
        )*
    };
}

#[macro_export]
macro_rules! define_binary_op_trait {
    ($trait_name:ident, $required_trait:ident, $name:ident, $operator:tt; $($default_dtypes:ty),*) => {
        define_binary_op_trait!($trait_name, $required_trait, $name, $operator);
        impl_default_binary_op_trait!($trait_name, $($default_dtypes),*);
    };

    ($trait_name:ident, $required_trait:ident, $name:ident, $operator:tt) => {
        paste! {
            pub(crate) trait $trait_name: $required_trait<Output=Self> + Sized + Copy {
                unsafe fn [<$name _stride_n_0>](lhs: *const Self, lhs_stride: usize,
                                                rhs: *const Self, dst: *mut Self, count: usize) {
                    // TODO SIMD kernel for this
                    Self::[<$name _stride_n_n>](lhs, lhs_stride, rhs, 0, dst, count)
                }

                unsafe fn [<$name _stride_n_1>](lhs: *const Self, lhs_stride: usize,
                                                rhs: *const Self, dst: *mut Self, count: usize) {
                    Self::[<$name _stride_n_n>](lhs, lhs_stride, rhs, 1, dst, count)
                }

                unsafe fn [<$name _stride_1_1>](lhs: *const Self, rhs: *const Self, dst: *mut Self, count: usize) {
                    Self::[<$name _stride_n_n>](lhs, 1, rhs, 1, dst, count)
                }

                unsafe fn [<$name _stride_n_n>](mut lhs: *const Self, lhs_stride: usize,
                                                mut rhs: *const Self, rhs_stride: usize,
                                                mut dst: *mut Self, mut count: usize) {
                    while count != 0 {
                        *dst = *lhs $operator *rhs;

                        count -= 1;
                        lhs = lhs.add(lhs_stride);
                        rhs = rhs.add(rhs_stride);
                        dst = dst.add(1);
                    }
                }

                unsafe fn [<$name _nonunif_0>](lhs: *const Self, lhs_shape: &[usize], lhs_stride: &[usize],
                                               rhs: *const Self,
                                               dst: *mut Self, count: usize) {
                    Self::[<$name _nonunif_n>](lhs, lhs_shape, lhs_stride, rhs, 0, dst, count)
                }

                unsafe fn [<$name _nonunif_1>](lhs: *const Self, lhs_shape: &[usize], lhs_stride: &[usize],
                                               rhs: *const Self,
                                               dst: *mut Self, count: usize) {
                    Self::[<$name _nonunif_n>](lhs, lhs_shape, lhs_stride, rhs, 1, dst, count)
                }

                unsafe fn [<$name _nonunif_n>](lhs: *const Self, lhs_shape: &[usize], lhs_stride: &[usize],
                                               mut rhs: *const Self, rhs_stride: usize,
                                               mut dst: *mut Self, mut count: usize) {
                    let mut lhs_indices = FlatIndexGenerator::from(lhs_shape, lhs_stride);

                    while count != 0 {
                        let lhs_index = lhs_indices.next().unwrap_unchecked();
                        *dst = *lhs.add(lhs_index) $operator *rhs;

                        count -= 1;
                        dst = dst.add(1);
                        rhs = rhs.add(rhs_stride);
                    }
                }

                unsafe fn [<$name _unspecialized>](lhs: *const Self, lhs_shape: &[usize], lhs_stride: &[usize],
                                                   rhs: *const Self, rhs_shape: &[usize], rhs_stride: &[usize],
                                                   mut dst: *mut Self) {
                    let lhs_indices = FlatIndexGenerator::from(lhs_shape, lhs_stride);
                    let rhs_indices = FlatIndexGenerator::from(rhs_shape, rhs_stride);

                    for (lhs_index, rhs_index) in lhs_indices.zip(rhs_indices) {
                        *dst = *lhs.add(lhs_index) $operator *rhs.add(rhs_index);
                        dst = dst.add(1);
                    }
                }

                unsafe fn $name(lhs: *const Self, lhs_stride: &[usize],
                                rhs: *const Self, rhs_stride: &[usize],
                                dst: *mut Self, shape: &[usize]) {
                    // special case for scalar operands
                    if lhs_stride.is_empty() && rhs_stride.is_empty() {
                        *dst = *lhs $operator *rhs;
                        return;
                    }

                    let (lhs_shape, lhs_stride) = collapse_to_uniform_stride(shape, &lhs_stride);
                    let (rhs_shape, rhs_stride) = collapse_to_uniform_stride(shape, &rhs_stride);

                    let lhs_dims = lhs_shape.len();
                    let rhs_dims = rhs_shape.len();

                    let lhs_inner_stride = lhs_stride[lhs_dims - 1];
                    let rhs_inner_stride = rhs_stride[rhs_dims - 1];

                    if lhs_dims == 1 && rhs_dims == 1 { // both operands have a uniform stride

                        // one operand is a scalar
                        if rhs_inner_stride == 0 {
                            return Self::[<$name _stride_n_0>](lhs, lhs_inner_stride, rhs, dst, lhs_shape[0]);
                        } else if lhs_inner_stride == 0 {
                            return Self::[<$name _stride_n_0>](rhs, rhs_inner_stride, lhs, dst, lhs_shape[0]);
                        }

                        // both operands are contiguous
                        if lhs_inner_stride == 1 && rhs_inner_stride == 1 {
                            return Self::[<$name _stride_1_1>](lhs, rhs, dst, lhs_shape[0]);
                        }

                        // neither element is contiguous
                        return Self::[<$name _stride_n_n>](lhs, lhs_inner_stride, rhs, rhs_inner_stride, dst, lhs_shape[0]);
                    }

                    // only 1 operand has a uniform stride
                    if rhs_dims == 1 && rhs_inner_stride == 0 {
                        return Self::[<$name _nonunif_0>](lhs, &lhs_shape, &lhs_stride,
                                                          rhs, dst, rhs_shape[0]);
                    } else if lhs_dims == 1 && lhs_inner_stride == 0 {
                        return Self::[<$name _nonunif_0>](rhs, &rhs_shape, &rhs_stride,
                                                          lhs, dst, lhs_shape[0]);
                    }

                    if rhs_dims == 1 && rhs_inner_stride == 1 {
                        return Self::[<$name _nonunif_1>](lhs, &lhs_shape, &lhs_stride,
                                                          rhs, dst, rhs_shape[0]);
                    } else if lhs_dims == 1 && lhs_inner_stride == 1 {
                        return Self::[<$name _nonunif_1>](rhs, &rhs_shape, &rhs_stride,
                                                          lhs, dst, lhs_shape[0]);
                    }

                    if rhs_dims == 1 {
                        return Self::[<$name _nonunif_n>](lhs, &lhs_shape, &lhs_stride,
                                                          rhs, rhs_inner_stride,
                                                          dst, rhs_shape[0]);
                    } else if lhs_dims == 1 {
                        return Self::[<$name _nonunif_n>](rhs, &rhs_shape, &rhs_stride,
                                                          lhs, lhs_inner_stride,
                                                          dst, lhs_shape[0]);
                    }

                    // unspecialized loop
                    Self::[<$name _unspecialized>](lhs, &lhs_shape, &lhs_stride,
                                                   rhs, &rhs_shape, &rhs_stride,
                                                   dst);
                }
            }
        }
    }
}

pub(crate) trait BinaryOpSub: Sub<Output=Self> + Sized + Copy {}
pub(crate) trait BinaryOpDiv: Div<Output=Self> + Sized + Copy {}
pub(crate) trait BinaryOpRem: Rem<Output=Self> + Sized + Copy {}
pub(crate) trait BinaryOpBitAnd: BitAnd<Output=Self> + Sized + Copy {}
pub(crate) trait BinaryOpBitOr: BitOr<Output=Self> + Sized + Copy {}
pub(crate) trait BinaryOpShl: Shl<Output=Self> + Sized + Copy {}
pub(crate) trait BinaryOpShr: Shr<Output=Self> + Sized + Copy {}

impl_default_binary_op_trait!(BinaryOpSub,
                              i8, i16, i32, i64, i128, isize,
                              u8, u16, u32, u64, u128, usize,
                              f32, f64);

impl_default_binary_op_trait!(BinaryOpRem,
                              i8, i16, i32, i64, i128, isize,
                              u8, u16, u32, u64, u128, usize,
                              f32, f64);

impl_default_binary_op_trait!(BinaryOpBitAnd,
                              i8, i16, i32, i64, i128, isize,
                              u8, u16, u32, u64, u128, usize);

impl_default_binary_op_trait!(BinaryOpBitOr,
                              i8, i16, i32, i64, i128, isize,
                              u8, u16, u32, u64, u128, usize);

impl_default_binary_op_trait!(BinaryOpShl,
                              i8, i16, i32, i64, i128, isize,
                              u8, u16, u32, u64, u128, usize);

impl_default_binary_op_trait!(BinaryOpShr,
                              i8, i16, i32, i64, i128, isize,
                              u8, u16, u32, u64, u128, usize);

impl_default_binary_op_trait!(BinaryOpDiv, f32, f64);
