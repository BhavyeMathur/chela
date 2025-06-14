//! # CPU & GPU Acceleration using vDSP, BLAS, LAPACK, CUDA, Arm64 NEON, etc
//!

pub mod simd_reduce_ops;
pub mod simd_sum_of_products;
pub mod simd_binary_ops;
pub mod simd_neg;

pub mod fill;

pub mod dot_product;

pub mod reduce_sum;
pub mod reduce_product;
pub mod reduce_min;
pub mod reduce_max;
pub mod reduce_min_magnitude;
pub mod reduce_max_magnitude;

pub mod binary_ops;
pub mod binary_op_add;
pub mod binary_op_mul;
pub mod binary_op_sub;
pub mod binary_op_div;
pub mod unary_ops;
