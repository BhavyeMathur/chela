use chela::*;

#[test]
fn test_autograd1() {
    let mut a = Tensor::scalar(2.0);
    let mut b = Tensor::scalar(3.0);

    a.set_requires_grad(true);
    b.set_requires_grad(true);
    
    let c = &a * &b;
    let d = &c * &b;
    d.backward(1.0);

    // d = ab^2 
    // dd/da = b^2
    // dd/db = 2ab

    assert_eq!(a.gradient(), Some(&b * &b));
    assert_eq!(b.gradient(), Some(&a * &b * 2.0));
}
