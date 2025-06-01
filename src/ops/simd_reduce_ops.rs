use crate::acceleration::simd::SIMD;

pub(crate) trait SIMDReduceOps: SIMD {
    #[cfg(neon_simd)]
    unsafe fn simd_sum_uniform(mut ptr: *const Self, mut count: usize, stride: usize) -> Self {
        let mut acc = Self::simd_dup(Self::zero());

        while count >= Self::LANES {
            let vec = if Self::LANES == 4 {
                let a = *ptr.add(0 * stride);
                let b = *ptr.add(1 * stride);
                let c = *ptr.add(2 * stride);
                let d = *ptr.add(3 * stride);

                Self::simd_from(&[a, b, c, d])
            }
            else if Self::LANES == 2 {
                let a = *ptr.add(0 * stride);
                let b = *ptr.add(1 * stride);

                Self::simd_from(&[a, b])
            } else {
                panic!("unimplemented SIMD sum uniform")
            };
            
            acc = Self::simd_add(acc, vec);

            count -= Self::LANES;
            ptr = ptr.add(stride * Self::LANES);
        }

        let mut output = Self::simd_sum(acc);

        for _ in 0..count {
            output += *ptr;
            ptr = ptr.add(stride);
        }

        output
    }

    #[cfg(neon_simd)]
    unsafe fn simd_sum_contiguous(mut ptr: *const Self, mut count: usize) -> Self {
        let mut acc = Self::simd_dup(Self::zero());

        while count >= 4 * Self::LANES {
            let a = Self::simd_load(ptr.add(0 * Self::LANES));
            let b = Self::simd_load(ptr.add(1 * Self::LANES));
            let c = Self::simd_load(ptr.add(2 * Self::LANES));
            let d = Self::simd_load(ptr.add(3 * Self::LANES));

            let ab = Self::simd_add(a, b);
            let cd = Self::simd_add(c, d);
            let abcd = Self::simd_add(ab, cd);

            count -= 4 * Self::LANES;
            ptr = ptr.add(4 * Self::LANES);
            
            acc = Self::simd_add(acc, abcd);
        }

        let mut output = Self::simd_sum(acc);
        for _ in 0..count {
            output += *ptr;
            ptr = ptr.add(1);
        }

        output
    }

    #[cfg(neon_simd)]
    unsafe fn simd_product_uniform(mut ptr: *const Self, mut count: usize, stride: usize) -> Self {
        let mut acc = Self::simd_dup(Self::one());

        while count >= Self::LANES {
            let vec = if Self::LANES == 4 {
                let a = *ptr.add(0 * stride);
                let b = *ptr.add(1 * stride);
                let c = *ptr.add(2 * stride);
                let d = *ptr.add(3 * stride);

                Self::simd_from(&[a, b, c, d])
            }
            else if Self::LANES == 2 {
                let a = *ptr.add(0 * stride);
                let b = *ptr.add(1 * stride);

                Self::simd_from(&[a, b])
            } else {
                   panic!("unimplemented SIMD product uniform") 
            };
            
            acc = Self::simd_mul(acc, vec);

            count -= Self::LANES;
            ptr = ptr.add(stride * Self::LANES);
        }

        let mut output = Self::simd_prod(acc);

        for _ in 0..count {
            output *= *ptr;
            ptr = ptr.add(stride);
        }

        output
    }

    #[cfg(neon_simd)]
    unsafe fn simd_product_contiguous(mut ptr: *const Self, mut count: usize) -> Self {
        let mut output = Self::one();

        while count >= 4 * Self::LANES {
            let a = Self::simd_load(ptr.add(0 * Self::LANES));
            let b = Self::simd_load(ptr.add(1 * Self::LANES));
            let c = Self::simd_load(ptr.add(2 * Self::LANES));
            let d = Self::simd_load(ptr.add(3 * Self::LANES));

            let ab = Self::simd_mul(a, b);
            let cd = Self::simd_mul(c, d);
            let abcd = Self::simd_mul(ab, cd);

            count -= 4 * Self::LANES;
            ptr = ptr.add(4 * Self::LANES);

            output *= Self::simd_prod(abcd);
        }

        for _ in 0..count {
            output *= *ptr;
            ptr = ptr.add(1);
        }

        output
    }
}

impl<T: SIMD> SIMDReduceOps for T {}
