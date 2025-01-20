use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use crate::dtype::RawDataType;
use crate::tensor::Tensor;

impl<T: RawDataType> Tensor<'_, T> {
    unsafe fn add(t1: &Tensor<T>, t2: &Tensor<T>) -> Tensor<T>{
        assert_eq!(t1.shape(), t2.shape(),
        "Tensor dimensions are not the same");
        let data: Vec<T> = vec![0; t1.shape().product()];
        // let mut idx = 0;

        for i in 0..data.len() {
            data[i] = t1.ptr.as_ptr().offset(i) + t2.ptr.as_ptr().offset(i);
        }

        let mut data = ManuallyDrop::new(data);

        Self{
            ptr: NonNull::new_unchecked(data.as_mut_ptr()),
            len: data.len(),
            capacity: data.capacity(),
            shape: t1.shape().clone(),
            stride: t1.stride().clone(),
            flags: (),
            _marker: Default::default(),
        }
    }
}