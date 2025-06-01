use crate::acceleration::simd::SIMD;

pub(crate) trait SIMDReduceOps: SIMD {
    #[cfg(neon_simd)]
    unsafe fn simd_sum_contiguous(mut ptr: *const Self, mut count: usize) -> Self {
        let mut output = Self::zero();

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

            output += Self::simd_sum(abcd);
        }

        for _ in 0..count {
            output += *ptr;
            ptr = ptr.add(1);
        }

        output
    }
}

impl<T: SIMD> SIMDReduceOps for T {}
