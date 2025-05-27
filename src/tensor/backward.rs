use crate::gradient_function::GradientFunction;
use crate::into_gradient::IntoTensor;
use crate::{FloatDataType, Tensor, TensorMethods};

impl<'a, T: FloatDataType> Tensor<'a, T> {
    pub(crate) fn get_grad_fn(&self) -> GradientFunction<T> {
        self.grad_fn.clone()
    }

    pub fn gradient(&self) -> Option<Tensor<T>> {
        self.grad_fn.borrow().gradient()
    }

    pub fn zero_gradient(&self) {
        match self.gradient() {
            None => {},
            Some(mut grad) => { grad.zero() }
        }
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