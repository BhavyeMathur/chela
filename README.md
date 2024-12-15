# Chela

A Rust library for linear algebra and machine learning!

## Changelog

### Dec 14, 2024

- Added a flat Iterator for `Tensor` and `TensorView`
- `clone()` & `flatten()` methods

### Dec 13, 2024

- Full `TensorBase` slicing support!
- `squeeze()` & `unsqueeze()` methods

### Dec 12, 2024

- Added `TensorView` constructors
- Implemented `Index<usize>` trait for `DataBuffer`
- Added `slice_along()` method for `Tensor`

### Dec 11, 2024

- Added `stride()` & `ndims()` methods for `Tensor`
- `Tensor` now stores `stride` & `shape` as `Vec<usize>`
- Implemented `Index` trait for `Tensor` & `DataOwned`

### Dec 10, 2024

- Optimised `Tensor` constructor for nested arrays using unsafe copy
- `Clone` and `Debug` traits for `Tensor`
- `TensorBase<T, N>` now stores its ndims as generic constant `N` using the `Nested<N>` trait & stores `shape` as an array

### Dec 9, 2024

- Created `Tensor` struct & constructor
- Wrote first tests for `Tensor` and `DataOwned`
