//! # Multidimensional Tensors with Dynamic Automatic Differentiation
//!
//! The `Tensor` API is nearly identical to `NdArray` with the following differences:
//! 1. Only floating point (`f32`, `f64`) types are supported
//! 2. Operations without autograd implemented are omitted
//!
//! `Tensors` allow us to perform dynamic automatic differentiation which is independent of
//! control flow. This allows us to find matrix derivatives even with complicated data paths.
//!
//! ```rust
//! # use redstone_ml::*;
//!  let mut a = Tensor::new([[7.5, 12.0], [5.0, 6.25]]);
//!  let mut b = Tensor::new([0.5, -2.0]);
//!  let c = Tensor::scalar(10.0);
//!
//!  a.set_requires_grad(true);
//!  b.set_requires_grad(true);
//!
//!  let matrix_2x2 = (&a / &b) * (c + 5.0);
//!  let result = matrix_2x2.matmul(&b);
//!  result.backward();
//!
//!  println!("{:?}", a.gradient().unwrap());
//!  println!("{:?}", b.gradient().unwrap());
//! ```
//!
//! You can also use `backwards_with(grad: NdArray<T>)` to find the gradient with a custom input.
//!

pub mod methods;
pub mod ops;
pub mod constructors;
pub mod equals;
pub mod autograd;
pub mod print;
pub mod matrix_ops;
pub mod reshape;

use std::marker::PhantomData;
use std::rc::Rc;
use crate::gradient_function::GradientFunction;
use crate::ndarray::flags::NdArrayFlags;
use crate::{NdArray, TensorDataType};

pub struct Tensor<'a, T: TensorDataType> {
    array: Rc<NdArray<'static, T>>,

    pub(super) flags: NdArrayFlags,
    pub(super) grad_fn: GradientFunction<T>,

    _marker: PhantomData<&'a T>,
}
