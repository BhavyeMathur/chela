use crate::gradient_function::GradientFunction;
use crate::into_gradient::IntoTensor;
use crate::{FloatDataType, Tensor, TensorMethods};

impl<'a, T: FloatDataType> Tensor<'a, T> {
    pub(crate) fn get_grad_fn(&'a self) -> GradientFunction<T> {
        self.grad_fn.clone()
    }

    pub fn gradient(&'a self) -> Option<Tensor<'a, T>> {
        unsafe { (*self.grad_fn.as_ptr()).gradient() }
    }

    pub fn zero_gradient(&self) {
        self.grad_fn.borrow_mut().zero();
    }

    pub fn backward_with(&self, gradient: impl IntoTensor<'a, T>) {
        let gradient = gradient.as_tensor();
        assert_eq!(gradient.shape(), self.shape());

        self.grad_fn.borrow_mut().backward(&gradient);
    }

    pub fn backward(&self) {
        self.backward_with(Tensor::ones(self.shape()))
    }
}