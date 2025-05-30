use crate::dtype::RawDataType;
use crate::NdArray;

impl<T: RawDataType> PartialEq<NdArray<'_, T>> for NdArray<'_, T> {
    fn eq(&self, other: &NdArray<T>) -> bool {
        &self == other
    }
}

impl<T: RawDataType> PartialEq<NdArray<'_, T>> for &NdArray<'_, T> {
    fn eq(&self, other: &NdArray<T>) -> bool {
        if self.shape != other.shape {
            return false;
        }
        self.flatiter().zip(other.flatiter()).all(|(a, b)| a == b.into())
    }
}

impl<T: RawDataType> PartialEq<&NdArray<'_, T>> for NdArray<'_, T> {
    fn eq(&self, other: &&NdArray<T>) -> bool {
        if self.shape != other.shape {
            return false;
        }
        self.flatiter().zip(other.flatiter()).all(|(a, b)| a == b.into())
    }
}
