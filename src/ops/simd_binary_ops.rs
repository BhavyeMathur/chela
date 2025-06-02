use paste::paste;
use crate::acceleration::simd::Simd;

macro_rules! simd_operation_stride_1_1 {
    ($name:ident, $simd_op:ident, $operator:tt) => {
        paste! {
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
        }
    };
}

pub(crate) trait SimdBinaryOps: Simd {
    simd_operation_stride_1_1!(add, simd_add, +);
    simd_operation_stride_1_1!(mul, simd_mul, *);
}

impl<T: Simd> SimdBinaryOps for T {}
