use crate::{NumericDataType, RawDataType, Tensor};

pub(crate) trait GradientFunction<T: RawDataType> {
    fn backward(&mut self, grad: &Tensor<T>);
}

struct AccumulateGrad<'a, T: NumericDataType> {
    tensor: &'a mut Tensor<'a, T>
}

struct AddBackwards<T: NumericDataType> {
    next_functions: Vec<Box<dyn GradientFunction<T>>>
}

impl<T: NumericDataType> GradientFunction<T> for AddBackwards<T> {
    fn backward(&mut self, grad: &Tensor<T>) {
        for func in self.next_functions.iter_mut() {
            func.backward(grad);
        }
    }
}

impl<'a, T: NumericDataType> GradientFunction<T> for AccumulateGrad<'a, T> {
    fn backward(&mut self, grad: &Tensor<T>) {
        self.tensor.set_grad(grad);
    }
}
