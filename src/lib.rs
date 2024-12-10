mod device;
mod tensor;

use tensor::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor() {
        let _arr = Tensor::from_vector(vec![0, 50, 100]);
        let _arr = Tensor::from_array([500, 50, 100]);

        let _arr = Tensor::from_vector(vec![vec![50], vec![50], vec![50]]);
        let _arr = Tensor::from_vector(vec![vec![vec![50]], vec![vec![50]], vec![vec![50]]]);
    }

    #[test]
    #[should_panic]
    fn assert_inhomogeneous_vector_error1() {
        Tensor::from_vector(vec![vec![50, 50], vec![50]]);
    }

    #[test]
    #[should_panic]
    fn assert_inhomogeneous_vector_error2() {
        Tensor::from_vector(vec![vec![vec![50, 50]], vec![vec![50]], vec![vec![50]]]);
    }
}
