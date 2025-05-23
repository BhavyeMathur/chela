pub(crate) fn transpose_2d_array<T: Default + Copy, const N: usize, const M: usize>(array: [[T; N]; M]) -> [[T; M]; N] {
    let mut result = [[T::default(); M]; N];

    for i in 0..M {
        for j in 0..N {
            result[j][i] = array[i][j];
        }
    }

    result
}


pub(crate) fn permute_array<T: Clone>(arr: &mut [T], permutation: &[usize]) {
    assert_eq!(arr.len(), permutation.len(), "Length mismatch between array and permutation");

    let original = arr.to_vec();
    for (i, &src_idx) in permutation.iter().enumerate() {
        arr[i] = original[src_idx].clone();
    }
}
