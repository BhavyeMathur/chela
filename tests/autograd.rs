use chela::*;

#[test]
fn test_autograd_sum() {
    let mut a = Tensor::scalar(2.0);
    let b = Tensor::scalar(3.0);

    a.set_requires_grad(true);
    
    let mut c = &a * b;
    c.backward(1.0);
    
    println!("{:?}", a.gradient())
}
