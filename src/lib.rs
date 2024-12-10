mod tensor;

use tensor::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vector() {
        Tensor::from_vector(vec![0, 50, 100]);
        Tensor::from_vector(vec![vec![50], vec![50], vec![50]]);
        Tensor::from_vector(vec![vec![vec![50]], vec![vec![50]], vec![vec![50]]]);
    }

    #[test]
    fn from_array() {
        Tensor::from_array([500, 50, 100]);
        Tensor::from_array([[500], [50], [100]]);
        Tensor::from_array([[[500], [50], [30]], [[50], [0], [0]], [[100], [10], [20]]]);
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
