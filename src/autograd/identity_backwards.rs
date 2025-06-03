use crate::gradient_function::GradientFunction;
use crate::{FloatDataType, Tensor};

pub(crate) struct IdentityBackwards {}

impl IdentityBackwards {
    pub(crate) fn new<T: FloatDataType>(tensor: &Tensor<T>) -> GradientFunction<T> {
        tensor.grad_fn()
    }
}
