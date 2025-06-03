use std::fmt::Display;
use paste::paste;
use crate::acceleration::simd::Simd;

macro_rules! simd_elementwise_operations {
    ($name:ident, $simd_op:ident, $operator:tt) => {
        paste! {
            #[cfg(neon_simd)]
            unsafe fn [<simd_ $name _stride_0_1>](lhs: *const Self, mut rhs: *const Self, mut dst: *mut Self, mut count: usize) {
                let a = Self::simd_from_constant(*lhs);

                while count >= 4 * Self::LANES {
                    let b0 = Self::simd_load(rhs.add(0 * Self::LANES));
                    let b1 = Self::simd_load(rhs.add(1 * Self::LANES));
                    let b2 = Self::simd_load(rhs.add(2 * Self::LANES));
                    let b3 = Self::simd_load(rhs.add(3 * Self::LANES));

                    let ab0 = Self::$simd_op(a, b0);
                    let ab1 = Self::$simd_op(a, b1);
                    let ab2 = Self::$simd_op(a, b2);
                    let ab3 = Self::$simd_op(a, b3);

                    Self::simd_store(dst.add(0 * Self::LANES), ab0);
                    Self::simd_store(dst.add(1 * Self::LANES), ab1);
                    Self::simd_store(dst.add(2 * Self::LANES), ab2);
                    Self::simd_store(dst.add(3 * Self::LANES), ab3);

                    count -= 4 * Self::LANES;
                    rhs = rhs.add(4 * Self::LANES);
                    dst = dst.add(4 * Self::LANES);
                }

                while count != 0 {
                    *dst = *lhs $operator *rhs;

                    count -= 1;
                    rhs = rhs.add(1);
                    dst = dst.add(1);
                }
            }

            #[cfg(neon_simd)]
            unsafe fn [<simd_ $name _stride_1_0>](mut lhs: *const Self, rhs: *const Self, mut dst: *mut Self, mut count: usize) {
                let b = Self::simd_from_constant(*rhs);

                while count >= 4 * Self::LANES {
                    let a0 = Self::simd_load(lhs.add(0 * Self::LANES));
                    let a1 = Self::simd_load(lhs.add(1 * Self::LANES));
                    let a2 = Self::simd_load(lhs.add(2 * Self::LANES));
                    let a3 = Self::simd_load(lhs.add(3 * Self::LANES));

                    let ab0 = Self::$simd_op(a0, b);
                    let ab1 = Self::$simd_op(a1, b);
                    let ab2 = Self::$simd_op(a2, b);
                    let ab3 = Self::$simd_op(a3, b);

                    Self::simd_store(dst.add(0 * Self::LANES), ab0);
                    Self::simd_store(dst.add(1 * Self::LANES), ab1);
                    Self::simd_store(dst.add(2 * Self::LANES), ab2);
                    Self::simd_store(dst.add(3 * Self::LANES), ab3);

                    count -= 4 * Self::LANES;
                    lhs = lhs.add(4 * Self::LANES);
                    dst = dst.add(4 * Self::LANES);
                }

                while count != 0 {
                    *dst = *lhs $operator *rhs;

                    count -= 1;
                    lhs = lhs.add(1);
                    dst = dst.add(1);
                }
            }

            #[cfg(neon_simd)]
            unsafe fn [<simd_ $name _stride_1_1>](mut lhs: *const Self, mut rhs: *const Self, mut dst: *mut Self, mut count: usize) {
                while count >= 4 * Self::LANES {
                    let a0 = Self::simd_load(lhs.add(0 * Self::LANES));
                    let b0 = Self::simd_load(rhs.add(0 * Self::LANES));

                    let a1 = Self::simd_load(lhs.add(1 * Self::LANES));
                    let b1 = Self::simd_load(rhs.add(1 * Self::LANES));

                    let a2 = Self::simd_load(lhs.add(2 * Self::LANES));
                    let b2 = Self::simd_load(rhs.add(2 * Self::LANES));

                    let a3 = Self::simd_load(lhs.add(3 * Self::LANES));
                    let b3 = Self::simd_load(rhs.add(3 * Self::LANES));

                    let ab0 = Self::$simd_op(a0, b0);
                    let ab1 = Self::$simd_op(a1, b1);
                    let ab2 = Self::$simd_op(a2, b2);
                    let ab3 = Self::$simd_op(a3, b3);

                    Self::simd_store(dst.add(0 * Self::LANES), ab0);
                    Self::simd_store(dst.add(1 * Self::LANES), ab1);
                    Self::simd_store(dst.add(2 * Self::LANES), ab2);
                    Self::simd_store(dst.add(3 * Self::LANES), ab3);

                    count -= 4 * Self::LANES;
                    lhs = lhs.add(4 * Self::LANES);
                    rhs = rhs.add(4 * Self::LANES);
                    dst = dst.add(4 * Self::LANES);
                }

                while count != 0 {
                    *dst = *lhs $operator *rhs;

                    count -= 1;
                    lhs = lhs.add(1);
                    rhs = rhs.add(1);
                    dst = dst.add(1);
                }
            }

            #[cfg(neon_simd)]
            unsafe fn [<simd_ $name _stride_n_0>](mut lhs: *const Self, lhs_stride: usize, rhs: *const Self, mut dst: *mut Self, mut count: usize) {
                 let b = Self::simd_from_constant(*rhs);

                 while count >= 4 * Self::LANES {
                    let a0 = Self::simd_vec_from_stride(lhs.add(0 * lhs_stride * Self::LANES), lhs_stride);
                    let a1 = Self::simd_vec_from_stride(lhs.add(1 * lhs_stride * Self::LANES), lhs_stride);
                    let a2 = Self::simd_vec_from_stride(lhs.add(2 * lhs_stride * Self::LANES), lhs_stride);
                    let a3 = Self::simd_vec_from_stride(lhs.add(3 * lhs_stride * Self::LANES), lhs_stride);

                    let ab0 = Self::$simd_op(a0, b);
                    let ab1 = Self::$simd_op(a1, b);
                    let ab2 = Self::$simd_op(a2, b);
                    let ab3 = Self::$simd_op(a3, b);

                    Self::simd_store(dst.add(0 * Self::LANES), ab0);
                    Self::simd_store(dst.add(1 * Self::LANES), ab1);
                    Self::simd_store(dst.add(2 * Self::LANES), ab2);
                    Self::simd_store(dst.add(3 * Self::LANES), ab3);

                    count -= 4 * Self::LANES;
                    lhs = lhs.add(4 * lhs_stride * Self::LANES);
                    dst = dst.add(4 * Self::LANES);
                }

                while count != 0 {
                    *dst = *lhs $operator *rhs;

                    count -= 1;
                    lhs = lhs.add(lhs_stride);
                    dst = dst.add(1);
                }
            }

            #[cfg(neon_simd)]
            unsafe fn [<simd_ $name _stride_0_n>](lhs: *const Self, mut rhs: *const Self, rhs_stride: usize, mut dst: *mut Self, mut count: usize) {
                 let a = Self::simd_from_constant(*lhs);

                 while count >= 4 * Self::LANES {
                    let b0 = Self::simd_vec_from_stride(rhs.add(0 * rhs_stride * Self::LANES), rhs_stride);
                    let b1 = Self::simd_vec_from_stride(rhs.add(1 * rhs_stride * Self::LANES), rhs_stride);
                    let b2 = Self::simd_vec_from_stride(rhs.add(2 * rhs_stride * Self::LANES), rhs_stride);
                    let b3 = Self::simd_vec_from_stride(rhs.add(3 * rhs_stride * Self::LANES), rhs_stride);

                    let ab0 = Self::$simd_op(a, b0);
                    let ab1 = Self::$simd_op(a, b1);
                    let ab2 = Self::$simd_op(a, b2);
                    let ab3 = Self::$simd_op(a, b3);

                    Self::simd_store(dst.add(0 * Self::LANES), ab0);
                    Self::simd_store(dst.add(1 * Self::LANES), ab1);
                    Self::simd_store(dst.add(2 * Self::LANES), ab2);
                    Self::simd_store(dst.add(3 * Self::LANES), ab3);

                    count -= 4 * Self::LANES;
                    rhs = rhs.add(4 * rhs_stride * Self::LANES);
                    dst = dst.add(4 * Self::LANES);
                }

                while count != 0 {
                    *dst = *lhs $operator *rhs;

                    count -= 1;
                    rhs = rhs.add(rhs_stride);
                    dst = dst.add(1);
                }
            }

            #[cfg(neon_simd)]
            unsafe fn [<simd_ $name _stride_n_1>](mut lhs: *const Self, lhs_stride: usize, mut rhs: *const Self, mut dst: *mut Self, mut count: usize) {
                while count >= 4 * Self::LANES {
                    let a0 = Self::simd_vec_from_stride(lhs.add(0 * lhs_stride * Self::LANES), lhs_stride);
                    let b0 = Self::simd_load(rhs.add(0 * Self::LANES));

                    let a1 = Self::simd_vec_from_stride(lhs.add(1 * lhs_stride * Self::LANES), lhs_stride);
                    let b1 = Self::simd_load(rhs.add(1 * Self::LANES));

                    let a2 = Self::simd_vec_from_stride(lhs.add(2 * lhs_stride * Self::LANES), lhs_stride);
                    let b2 = Self::simd_load(rhs.add(2 * Self::LANES));

                    let a3 = Self::simd_vec_from_stride(lhs.add(3 * lhs_stride * Self::LANES), lhs_stride);
                    let b3 = Self::simd_load(rhs.add(3 * Self::LANES));

                    let ab0 = Self::$simd_op(a0, b0);
                    let ab1 = Self::$simd_op(a1, b1);
                    let ab2 = Self::$simd_op(a2, b2);
                    let ab3 = Self::$simd_op(a3, b3);

                    Self::simd_store(dst.add(0 * Self::LANES), ab0);
                    Self::simd_store(dst.add(1 * Self::LANES), ab1);
                    Self::simd_store(dst.add(2 * Self::LANES), ab2);
                    Self::simd_store(dst.add(3 * Self::LANES), ab3);

                    count -= 4 * Self::LANES;
                    lhs = lhs.add(4 * lhs_stride * Self::LANES);
                    rhs = rhs.add(4 * Self::LANES);
                    dst = dst.add(4 * Self::LANES);
                }

                while count != 0 {
                    *dst = *lhs $operator *rhs;

                    count -= 1;
                    lhs = lhs.add(lhs_stride);
                    rhs = rhs.add(1);
                    dst = dst.add(1);
                }
            }

            #[cfg(neon_simd)]
            unsafe fn [<simd_ $name _stride_1_n>](mut lhs: *const Self, mut rhs: *const Self, rhs_stride: usize, mut dst: *mut Self, mut count: usize) {
                while count >= 4 * Self::LANES {
                    let a0 = Self::simd_load(lhs.add(0 * Self::LANES));
                    let b0 = Self::simd_vec_from_stride(rhs.add(0 * rhs_stride * Self::LANES), rhs_stride);

                    let a1 = Self::simd_load(lhs.add(1 * Self::LANES));
                    let b1 = Self::simd_vec_from_stride(rhs.add(1 * rhs_stride * Self::LANES), rhs_stride);

                    let a2 = Self::simd_load(lhs.add(2 * Self::LANES));
                    let b2 = Self::simd_vec_from_stride(rhs.add(2 * rhs_stride * Self::LANES), rhs_stride);

                    let a3 = Self::simd_load(lhs.add(3 * Self::LANES));
                    let b3 = Self::simd_vec_from_stride(rhs.add(3 * rhs_stride * Self::LANES), rhs_stride);

                    let ab0 = Self::$simd_op(a0, b0);
                    let ab1 = Self::$simd_op(a1, b1);
                    let ab2 = Self::$simd_op(a2, b2);
                    let ab3 = Self::$simd_op(a3, b3);

                    Self::simd_store(dst.add(0 * Self::LANES), ab0);
                    Self::simd_store(dst.add(1 * Self::LANES), ab1);
                    Self::simd_store(dst.add(2 * Self::LANES), ab2);
                    Self::simd_store(dst.add(3 * Self::LANES), ab3);

                    count -= 4 * Self::LANES;
                    lhs = lhs.add(4 * Self::LANES);
                    rhs = rhs.add(4 * rhs_stride * Self::LANES);
                    dst = dst.add(4 * Self::LANES);
                }

                while count != 0 {
                    *dst = *lhs $operator *rhs;

                    count -= 1;
                    lhs = lhs.add(1);
                    rhs = rhs.add(rhs_stride);
                    dst = dst.add(1);
                }
            }

            #[cfg(neon_simd)]
            unsafe fn [<simd_ $name _stride_n_n>](mut lhs: *const Self, lhs_stride: usize,
                                                  mut rhs: *const Self, rhs_stride: usize,
                                                  mut dst: *mut Self, mut count: usize) {
                while count >= 4 * Self::LANES {
                    let a0 = Self::simd_vec_from_stride(lhs.add(0 * lhs_stride * Self::LANES), lhs_stride);
                    let b0 = Self::simd_vec_from_stride(rhs.add(0 * rhs_stride * Self::LANES), rhs_stride);

                    let a1 = Self::simd_vec_from_stride(lhs.add(1 * lhs_stride * Self::LANES), lhs_stride);
                    let b1 = Self::simd_vec_from_stride(rhs.add(1 * rhs_stride * Self::LANES), rhs_stride);

                    let a2 = Self::simd_vec_from_stride(lhs.add(2 * lhs_stride * Self::LANES), lhs_stride);
                    let b2 = Self::simd_vec_from_stride(rhs.add(2 * rhs_stride * Self::LANES), rhs_stride);

                    let a3 = Self::simd_vec_from_stride(lhs.add(3 * lhs_stride * Self::LANES), lhs_stride);
                    let b3 = Self::simd_vec_from_stride(rhs.add(3 * rhs_stride * Self::LANES), rhs_stride);

                    let ab0 = Self::$simd_op(a0, b0);
                    let ab1 = Self::$simd_op(a1, b1);
                    let ab2 = Self::$simd_op(a2, b2);
                    let ab3 = Self::$simd_op(a3, b3);

                    Self::simd_store(dst.add(0 * Self::LANES), ab0);
                    Self::simd_store(dst.add(1 * Self::LANES), ab1);
                    Self::simd_store(dst.add(2 * Self::LANES), ab2);
                    Self::simd_store(dst.add(3 * Self::LANES), ab3);

                    count -= 4 * Self::LANES;
                    lhs = lhs.add(4 * lhs_stride * Self::LANES);
                    rhs = rhs.add(4 * rhs_stride * Self::LANES);
                    dst = dst.add(4 * Self::LANES);
                }

                while count != 0 {
                    *dst = *lhs $operator *rhs;

                    count -= 1;
                    lhs = lhs.add(lhs_stride);
                    rhs = rhs.add(rhs_stride);
                    dst = dst.add(1);
                }
            }
        }
    };
}

pub(crate) trait SimdBinaryOps: Simd + Display {
    simd_elementwise_operations!(add, simd_add, +);
    simd_elementwise_operations!(sub, simd_sub, -);
    simd_elementwise_operations!(mul, simd_mul, *);
    simd_elementwise_operations!(div, simd_div, /);
}

impl<T: Simd + Display> SimdBinaryOps for T {}

#[macro_export]
macro_rules! simd_binary_op_specializations {
    ($name: ident) => {
        paste! {
            #[cfg(neon_simd)]
            unsafe fn [<$name _stride_0_1>](lhs: *const Self, rhs: *const Self,
                                            dst: *mut Self, count: usize) {
                use $crate::ops::simd_binary_ops::SimdBinaryOps;
                Self::[<simd_ $name _stride_0_1>](lhs, rhs, dst, count);
            }

            #[cfg(neon_simd)]
            unsafe fn [<$name _stride_1_0>](lhs: *const Self, rhs: *const Self,
                                            dst: *mut Self, count: usize) {
                use $crate::ops::simd_binary_ops::SimdBinaryOps;
                Self::[<simd_ $name _stride_1_0>](lhs, rhs, dst, count);
            }

            #[cfg(neon_simd)]
            unsafe fn [<$name _stride_0_n>](lhs: *const Self,
                                            rhs: *const Self, rhs_stride: usize,
                                            dst: *mut Self, count: usize) {
                use $crate::ops::simd_binary_ops::SimdBinaryOps;
                Self::[<simd_ $name _stride_0_n>](lhs, rhs, rhs_stride, dst, count);
            }

            #[cfg(neon_simd)]
            unsafe fn [<$name _stride_n_0>](lhs: *const Self, lhs_stride: usize,
                                            rhs: *const Self,
                                            dst: *mut Self, count: usize) {
                use $crate::ops::simd_binary_ops::SimdBinaryOps;
                Self::[<simd_ $name _stride_n_0>](lhs, lhs_stride, rhs, dst, count);
            }

            #[cfg(neon_simd)]
            unsafe fn [<$name _stride_1_1>](lhs: *const Self, rhs: *const Self,
                                            dst: *mut Self, count: usize) {
                use $crate::ops::simd_binary_ops::SimdBinaryOps;
                Self::[<simd_ $name _stride_1_1>](lhs, rhs, dst, count);
            }

            #[cfg(neon_simd)]
            unsafe fn [<$name _stride_1_n>](lhs: *const Self,
                                            rhs: *const Self, rhs_stride: usize,
                                            dst: *mut Self, count: usize) {
                use $crate::ops::simd_binary_ops::SimdBinaryOps;
                Self::[<simd_ $name _stride_1_n>](lhs, rhs, rhs_stride, dst, count);
            }

            #[cfg(neon_simd)]
            unsafe fn [<$name _stride_n_1>](lhs: *const Self, lhs_stride: usize,
                                            rhs: *const Self,
                                            dst: *mut Self, count: usize) {
                use $crate::ops::simd_binary_ops::SimdBinaryOps;
                Self::[<simd_ $name _stride_n_1>](lhs, lhs_stride, rhs, dst, count);
            }

            #[cfg(neon_simd)]
            unsafe fn [<$name _stride_n_n>](lhs: *const Self, lhs_stride: usize,
                                            rhs: *const Self, rhs_stride: usize,
                                            dst: *mut Self, count: usize) {
                use $crate::ops::simd_binary_ops::SimdBinaryOps;
                Self::[<simd_ $name _stride_n_n>](lhs, lhs_stride, rhs, rhs_stride, dst, count);
            }
        }
    };
}
