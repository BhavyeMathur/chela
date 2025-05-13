use chela::*;

#[test]
fn full_i32() {
    let a = Tensor::full(3, [2, 3]);
    assert_eq!(a.shape(), &[2, 3]);
    assert_eq!(a.stride(), &[3, 1]);
    assert!(a.flatiter().all(|x| x == 3));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn full_f64() {
    let a = Tensor::full(3.2, [4, 6, 2]);
    assert_eq!(a.shape(), &[4, 6, 2]);
    assert!(a.flatiter().all(|x| x == 3.2));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn full_bool() {
    let a: Tensor<bool> = Tensor::full(true, vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    assert!(a.flatiter().all(|x| x == true));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn ones_u8() {
    let a: Tensor<u8> = Tensor::ones([3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    assert!(a.flatiter().all(|x| x == 1));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn ones_i32() {
    let a: Tensor<i32> = Tensor::ones(vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    assert!(a.flatiter().all(|x| x == 1));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn ones_1d() {
    let a: Tensor<u8> = Tensor::ones([4]);
    assert_eq!(a.shape(), &[4]);
    assert!(a.flatiter().all(|x| x == 1));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn ones_f64() {
    let a: Tensor<f64> = Tensor::ones(vec![4]);
    assert_eq!(a.shape(), &[4]);
    assert!(a.flatiter().all(|x| x == 1.0));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn ones_bool() {
    let a: Tensor<bool> = Tensor::ones(vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    assert!(a.flatiter().all(|x| x == true));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn zeroes_u8() {
    let a: Tensor<u8> = Tensor::zeros([3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    assert!(a.flatiter().all(|x| x == 0));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn zeroes_i32() {
    let a: Tensor<i32> = Tensor::zeros(vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    assert!(a.flatiter().all(|x| x == 0));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn zeroes_1d() {
    let a: Tensor<u8> = Tensor::zeros([4]);
    assert_eq!(a.shape(), &[4]);
    assert!(a.flatiter().all(|x| x == 0));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn zeroes_f64() {
    let a: Tensor<f64> = Tensor::zeros(vec![4]);
    assert_eq!(a.shape(), &[4]);
    assert!(a.flatiter().all(|x| x == 0.0));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn zeroes_bool() {
    let a: Tensor<bool> = Tensor::zeros(vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    assert!(a.flatiter().all(|x| x == false));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn random_normal_f32() {
    let a: Tensor<f32> = Tensor::randn(vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert!(!a.is_view());
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn random_normal_f64() {
    let a: Tensor<f64> = Tensor::randn(vec![3, 5, 3]);
    let _: Vec<_> = a.flatiter().collect();
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert!(!a.is_view());
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn random_uniform_f64() {
    let a: Tensor<f64> = Tensor::rand(vec![2, 3]);
    let _: Vec<_> = a.flatiter().collect();
    assert_eq!(a.shape(), &[2, 3]);
    assert!(!a.is_view());
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn random_uniform_f32() {
    let a: Tensor<f32> = Tensor::rand(vec![2, 3, 6]);
    let _: Vec<_> = a.flatiter().collect();
    assert_eq!(a.shape(), &[2, 3, 6]);
    assert!(!a.is_view());
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn scalar_f32() {
    let a: Tensor<f32> = Tensor::scalar(5.0);
    let _: Vec<_> = a.flatiter().collect();
    assert_eq!(a.shape(), &[]);
    assert!(!a.is_view());
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(0));
}
