mod tensor;
mod traits;

#[cfg(test)]
mod tests {
    use crate::tensor::data_owned::*;

    #[test]
    fn from_vector() {
        let arr = DataOwned::from_vector(vec![0, 50, 100]);
        assert_eq!(arr.len(), &3);
        // assert_eq!(arr.shape(), &vec![3]);

        let arr = DataOwned::from_vector(vec![vec![50], vec![50], vec![50]]);
        assert_eq!(arr.len(), &3);
        // assert_eq!(arr.shape(), &vec![3, 1]);

        let arr = DataOwned::from_vector(vec![vec![vec![50]], vec![vec![50]]]);
        assert_eq!(arr.len(), &2);
        // assert_eq!(arr.shape(), &vec![2, 1, 1]);

        let arr = DataOwned::from_vector(vec![vec![vec![50, 50, 50]], vec![vec![50, 50, 50]]]);
        assert_eq!(arr.len(), &6);
        // assert_eq!(arr.shape(), &vec![2, 1, 3]);
    }

    #[test]
    fn from_array() {
        let arr = DataOwned::from_array([500, 50, 100]);
        assert_eq!(arr.len(), &3);
        // assert_eq!(arr.shape(), &vec![3]);

        let arr = DataOwned::from_array([[500], [50], [100]]);
        assert_eq!(arr.len(), &3);
        // assert_eq!(arr.shape(), &vec![3, 1]);

        let arr = DataOwned::from_array([[[500], [50], [30]], [[50], [0], [0]]]);
        assert_eq!(arr.len(), &6);
        // assert_eq!(arr.shape(), &vec![2, 3, 1]);

        let arr = DataOwned::from_array([[[50, 50]], [[50, 50]]]);
        assert_eq!(arr.len(), &4);
        // assert_eq!(arr.shape(), &vec![2, 1, 3]);
    }

    #[test]
    #[should_panic]
    fn assert_inhomogeneous_vector_error1() {
        DataOwned::from_vector(vec![vec![50, 50], vec![50]]);
    }

    #[test]
    #[should_panic]
    fn assert_inhomogeneous_vector_error2() {
        DataOwned::from_vector(vec![vec![vec![50, 50]], vec![vec![50]], vec![vec![50]]]);
    }
}
