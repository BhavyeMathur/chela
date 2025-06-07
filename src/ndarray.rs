//! # N-dimensional Array for Linear Algebra & Tensor Computations
//!
//! An `NdArray` is a fixed-size multidimensional array container defined by its `shape`
//! and datatype. 1D (vectors) and 2D (matrices) arrays are often of special interest
//! and can be used in various linear algebra computations
//! including dot products, matrix products, batch matrix multiplications, einsums, and more.
//!
//! `NdArrays` can be iterated over (along configurable dimensions), reshaped, sliced, and indexed,
//! reduced, and more.
//!
//! This struct is heavily modeled after NumPy's `ndarray` and supports many of the same methods.
//!
//! Example:
//!
//! ```rust
//! use redstone_ml::*;
//!
//! let matrix_a = NdArray::new([[1, 3, 2], [-1, 0, -1]]); // shape [2, 3]
//! let matrix_b = NdArray::randint([3, 7], -5, 3);
//!
//! let matrix_view = matrix_b.slice_along(Axis(1), 0..2); // shape [3, 2]
//! let matrix_c = matrix_a.matmul(matrix_view);
//!
//! let result = matrix_c.sum();
//! ```
//!
//! ## NdArray Views & Lifetimes
//!
//! There are 2 ways we can create NdArray views: by borrowing or by consuming:
//! ```rust
//! # use redstone_ml::*;
//! let data = NdArray::<f64>::rand([9]);
//! let matrix = (&data).reshape([3, 3]); // by borrowing (data remains alive after)
//!
//! let data = NdArray::<f64>::rand([9]);
//! let matrix = data.reshape([3, 3]); // by consuming data
//! ```
//!
//! The consuming syntax allows us to chain operations without worrying about lifetimes
//! ```rust
//! # use redstone_ml::*;
//! // a reshaped and transposed random matrix
//! let matrix = NdArray::<f64>::rand([9]).reshape([3, 3]).T();
//! ```
//!
//! Operations like `reshape`, `view`, `diagonal`, `squeeze`, `unsqueeze`, `T`, `transpose`, and
//! `ravel` do not create new NdArrays by duplicating memory (which would be slow).
//! They always return `NdArray` views which share memory with the source `NdArray`.
//! `NdArray::clone()` or `NdArray::flatten()` can be used to duplicate the underlying `NdArray`.
//!
//! This means that all `NdArray` views have a lifetime at-most as long as the source `NdArray`.
//!
//! ## Linear Algebra, Broadcasting, and Reductions
//!
//! We currently support the core linear algebra operations including dot products,
//! matrix-vector and matrix-matrix multiplications, batched matrix multiplications, and trace.
//!
//! ```rust
//! # use redstone_ml::*;
//! # let matrix = NdArray::<f64>::randn([3, 3]);
//! # let matrix1 = NdArray::<f64>::randn([3, 3]);
//! # let matrix2 = NdArray::<f64>::randn([3, 3]);
//! # let batch_matrices1 = NdArray::<f64>::randn([2, 3, 3]);
//! # let batch_matrices2 = NdArray::<f64>::randn([2, 3, 3]);
//! # let vector = NdArray::<f64>::randn([3]);
//! # let vector1 = NdArray::<f64>::randn([3]);
//! # let vector2 = NdArray::<f64>::randn([3]);
//! vector1.dot(vector2);
//!
//! matrix.trace(); // also trace_along/offset_trace
//! matrix.diagonal(); // also diagonal_along/offset_diagonal
//! matrix.matmul(&vector);
//! matrix1.matmul(&matrix2);
//!
//! batch_matrices1.bmm(batch_matrices2);
//! 
//! // generic einsums 
//! einsum([&matrix1, &matrix2, &vector], (["ij", "kj", "i"], "ik"));
//! ```
//!
//! We can also perform various reductions including `sum`, `product`, `min`, `max`,
//! `min_magnitude`, and `max_magnitude`. Each of these is accelerated with various libraries
//! including vDSP, Arm64 NEON SIMD, and BLAS.
//!
//! ```rust
//! # use redstone_ml::*;
//! # let ndarray = NdArray::<f64>::zeros([5, 5, 5]);
//! let sum = ndarray.sum();
//! let sum_along = ndarray.sum_along([0, -1]); // sum along first and last axes
//! ```
//!
//! `NdArrays` can be used in arithmetic operations using the usual binary operators including
//! addition (`+`), subtraction (`-`), multiplication (`*`), division (`/`), remainder (`%`),
//! and bitwise operations (`&`, `|`, `<<`, `>>`).
//!
//! ```rust
//! # use redstone_ml::*;
//! # let arr1 = NdArray::<f64>::zeros([2, 2, 2]);
//! # let arr2 = NdArray::<f64>::zeros([2, 2, 2]);
//! let result = &arr1 + &arr2; // non-consuming
//! let result = &arr1 + arr2;  // consumes RHS
//! # let arr2 = NdArray::<f64>::zeros([2, 2, 2]);
//! let result = arr1 + arr2;   // consumes both
//! ```
//!
//! `NdArrays` are automatically broadcast using the exact same rules as NumPy
//! to perform efficient computations with different-dimensional (yet compatible) data.
//!
//! ## Slicing, Indexing, and Iterating
//!
//! Slicing and indexing an `NdArray` always return a view. This is how we can access various
//! elements of vectors, columns/rows of matrices, and more.
//!
//! ```rust
//! # use redstone_ml::*;
//! let arr = NdArray::<f32>::rand([2, 4, 3, 5]); // 4D NdArray
//! let slice1 = arr.slice(s![.., 0, ..=2]);      // use s! to specify a slice
//! let slice2 = arr.slice_along(Axis(-2), 0);    // 0th element along second-to-last axis
//! let el = arr[[0, 3, 2, 4]];
//! ```
//!
//! One can also iterate over an `NdArray` in various ways:
//! ```rust
//! # use redstone_ml::*;
//! # let arr = NdArray::<f32>::rand([2, 4, 3, 5]); // 4D NdArray
//! for subarray in arr.iter() { /* 4x3x5 subarrays */ }
//! for subarray in arr.iter_along(Axis(2)) { /* 2x4x5 subarrays */ }
//! for el in arr.flatiter() { /* element-wise iteration */ }
//! ```
//!
//! ## Other Constructors
//!
//! ```rust
//! # use redstone_ml::*;
//! let ndarray = NdArray::arange(0i32, 5); // [0, 1, 2, 3, 4]
//! let ndarray = NdArray::linspace(0f32, 1.0, 5); // [0.0, 0.25, 0.5, 0.75, 1.0]
//! ```
//!
//! ```rust
//! # use redstone_ml::*;
//! let ndarray = NdArray::full(5.0, [5, 4, 2]);
//! let falses = NdArray::<bool>::zeros([5, 4, 2]);
//! ```
//!
//! A scalar `NdArray` is dimensionless and contains a single value.
//! It is often the return value for reduction methods like `sum`, `product`, `min`, and `max`.
//! ```rust
//! # use redstone_ml::*;
//! let ten = NdArray::scalar(10u8);
//! ```
//!
//! In many cases, one desires randomized multidimensional arrays with a specified shape.
//! ```rust
//! # use redstone_ml::*;
//! let rand = NdArray::<f32>::randn([3, 4]);
//! let rand = NdArray::<f32>::rand([3, 4]);
//! let rand = NdArray::randint([3, 4], -5, 3);
//! ```


use std::marker::PhantomData;
use std::ptr::NonNull;

pub mod methods;

pub mod iterator;
pub use iterator::*;

pub mod reshape;

pub(crate) mod flags;
use flags::NdArrayFlags;

pub mod reduce;

pub mod constructors;
pub mod index_impl;
pub mod slice;
pub mod fill;
pub mod clone;
pub mod equals;
pub mod broadcast;
pub mod binary_ops;
pub mod astype;

mod print;
mod unary_ops;
mod assign_ops;

pub(crate) const MAX_DIMS: usize = 32;
pub(crate) const MAX_ARGS: usize = 16;

use crate::dtype::RawDataType;

pub struct NdArray<'a, T: RawDataType> {
    pub(crate) ptr: NonNull<T>,
    len: usize,
    capacity: usize,

    shape: Vec<usize>,
    stride: Vec<usize>,
    pub(crate) flags: NdArrayFlags,

    _marker: PhantomData<&'a T>,
}
