use crate::{RawDataType, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) type GradientFunction<T> = Rc<RefCell<dyn GradientFuncTrait<T>>>;

pub(crate) trait GradientFuncTrait<T: RawDataType> {
    fn backward(&mut self, grad: &Tensor<T>);
}

pub(crate) struct NoneBackwards {}

impl NoneBackwards {
    pub(crate) fn new<T: RawDataType>() -> GradientFunction<T> {
        Rc::new(RefCell::new(Self {}))
    }
}

impl<T: RawDataType> GradientFuncTrait<T> for NoneBackwards {
    fn backward(&mut self, _: &Tensor<T>) {}
}


pub(crate) struct AccumulateGrad<T: RawDataType> {
    tensor: *mut T
}


impl<T: RawDataType> AccumulateGrad<T> {
    pub(crate) fn new(tensor: *mut T) -> GradientFunction<T> {
        Rc::new(RefCell::new(Self {
            tensor
        }))
    }
}

impl<T: RawDataType> GradientFuncTrait<T> for AccumulateGrad<T> {
    fn backward(&mut self, grad: &Tensor<T>) {
        
    }
}
