pub(super) mod clone;
pub(super) mod data_owned;
pub(super) mod data_view;
pub(super) mod buffer;

pub(super) use crate::data_buffer::buffer::DataBuffer;
pub(super) use crate::data_buffer::data_owned::DataOwned;
pub(super) use crate::data_buffer::data_view::DataView;

use crate::tensor::dtype::RawDataType;

use std::ops::Index;

#[cfg(test)]
mod tests {
    use crate::data_buffer::DataOwned;

    #[test]
    fn from_vector() {
        let arr = DataOwned::from(vec![0, 50, 100]);
        assert_eq!(arr.len(), &3);

        let arr = DataOwned::from(vec![vec![50], vec![50], vec![50]]);
        assert_eq!(arr.len(), &3);

        let arr = DataOwned::from(vec![vec![vec![50]], vec![vec![50]]]);
        assert_eq!(arr.len(), &2);

        let arr = DataOwned::from(vec![vec![vec![50, 50, 50]], vec![vec![50, 50, 50]]]);
        assert_eq!(arr.len(), &6);
    }

    #[test]
    fn from_array() {
        let arr = DataOwned::from([500, 50, 100]);
        assert_eq!(arr.len(), &3);

        let arr = DataOwned::from([[500], [50], [100]]);
        assert_eq!(arr.len(), &3);

        let arr = DataOwned::from([[[500], [50], [30]], [[50], [0], [0]]]);
        assert_eq!(arr.len(), &6);

        let arr = DataOwned::from([[[50, 50]], [[50, 50]]]);
        assert_eq!(arr.len(), &4);
    }

    #[test]
    fn from_inhomogeneous_vector() {
        let arr = DataOwned::from(vec![vec![50, 50], vec![50]]);
        assert_eq!(arr.len(), &3);

        let arr = DataOwned::from(vec![vec![vec![50, 50]], vec![vec![50]], vec![vec![50]]]);
        assert_eq!(arr.len(), &4);
    }
}
