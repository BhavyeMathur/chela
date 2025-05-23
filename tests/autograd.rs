use chela::*;

#[test]
fn test_autograd_sum() {
    let mut a = Tensor::from([0f32, 2.0, 4.0]);
    let b = Tensor::from([0f32, 2.0, 4.0]);

    a.set_requires_grad(true);
    
    let mut c = a * b;
    c.backward(1.0);
    
    println!("{:?}", c.gradient())
}
