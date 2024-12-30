use crate::data_buffer::fill::Fill;
use crate::data_buffer::DataBuffer;
use crate::dtype::RawDataType;
use crate::TensorBase;

impl<B, T> TensorBase<B>
where
    B: DataBuffer<DType=T> + Fill<T>,
    T: RawDataType,
{
    pub fn fill(&mut self, value: T) {
        self.data.fill(value)
    }

    pub fn fill_naive(&mut self, value: T) {
        self.data.fill_naive(value)
    }
}

#[cfg(target_vendor = "apple")]
#[cfg(test)]
mod tests {
    use crate::{FlatIterator, Tensor};

    #[test]
    fn test_fill_f32() {
        let mut a: Tensor<f32> = Tensor::zeros([3, 5, 3]);

        assert!(a.flat_iter().all(|x| x == 0.0));
        a.fill(25.0);
        assert!(a.flat_iter().all(|x| x == 25.0));
    }

    #[test]
    fn test_fill_f64() {
        let mut a: Tensor<f64> = Tensor::zeros([15]);

        assert!(a.flat_iter().all(|x| x == 0.0));
        a.fill(20.0);
        assert!(a.flat_iter().all(|x| x == 20.0));
    }
}
