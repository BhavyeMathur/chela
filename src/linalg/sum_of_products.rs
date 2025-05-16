use crate::dtype::NumericDataType;

unsafe fn sum_of_products_generic<const N: usize, T: NumericDataType>(ptrs: [*const T; N],
                                                                      strides: [usize; N],
                                                                      count: usize) -> T {
    let mut sum = T::zero();

    let mut k = count;
    while k != 0 {
        k -= 1;
        sum += ptrs.iter().zip(strides.iter())
                   .map(|(ptr, stride)| *ptr.add(k * stride))
                   .product();
    }

    sum
}

unsafe fn sum_of_products_stride0_contig_outstride0_two<const N: usize, T: NumericDataType>(ptrs: [*const T; N],
                                                                                            _: [usize; N],
                                                                                            count: usize) -> T {
    let value0 = *ptrs[0];
    let data1 = std::slice::from_raw_parts(ptrs[1], count);
    value0 * data1.iter().copied().sum()
}


pub(super) fn get_sum_of_products_function<const N: usize, T: NumericDataType>(strides: [usize; N])
                                                                               -> unsafe fn([*const T; N], [usize; N], usize) -> T {
    if N == 2 {
        let mut code = if strides[0] == 0 { 0 } else { if strides[0] == 1 { 4 } else { 8 } };
        code += if strides[1] == 0 { 0 } else { if strides[1] == 1 { 2 } else { 8 } };
        // code += if strides[2] == 0 { 0 } else { if strides[2] == 1 { 1 } else { 8 } };  // NumPy stores the output's stride as element 2

        if code == 2 {
            return sum_of_products_stride0_contig_outstride0_two;
        }
    }

    sum_of_products_generic
}

// fn get_sum_of_products_function(int nop, int type_num, npy_intp itemsize, npy_intp const *fixed_strides)
// ->  {
//
// }
