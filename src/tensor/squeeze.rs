use crate::tensor::dtype::RawDataType;
use crate::tensor::Tensor;

// TODO implement for tensorview as well
// Squeeze for tensors
impl<T: RawDataType> Tensor<T>{
    pub fn squeeze_inplace(&mut self){
        self.stride.dedup();
        if self.shape[0] == 1 {
            self.stride.remove(0);
            println!("{:?}", self.stride);
        }
        self.shape.retain(|&i| i != 1);
    }

    pub fn unsqueeze_inplace(&mut self, axis: usize){
        assert!(axis <= self.shape.len(), "dimension out of bounds");
        self.shape.insert(axis, 1);
    }

    pub fn squeeze(self) -> Tensor<T> where T:RawDataType{
        self.data.
    }
}