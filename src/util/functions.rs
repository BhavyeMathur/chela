pub(crate) fn transpose_2d_array<T: Default + Copy, const N: usize, const M: usize>(array: [[T; N]; M]) -> [[T; M]; N] {
    let mut result = [[T::default(); M]; N];

    for (i, row) in array.iter().enumerate() {
        for (j, &item) in row.iter().enumerate() {
            result[j][i] = item;
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


/// Left pads a slice with a specified value until it reaches a specified length.
///
/// # Parameters
/// - `arr`: The slice to pad.
/// - `value`: The value to pad with.
/// - `n`: The desired length.
///
/// # Examples
///
/// ```ignore
/// let arr = vec![1, 2, 3];
/// let padded = pad(&arr, 0, 5);
/// assert_eq!(padded, vec![0, 0, 1, 2, 3]);
/// ```
pub(crate) fn pad<T: Copy>(arr: &[T], value: T, n: usize) -> Vec<T> {
    let mut new_arr = Vec::with_capacity(n);

    for _ in 0..(n - arr.len()) {
        new_arr.push(value);
    }
    new_arr.extend(arr);
    new_arr
}
