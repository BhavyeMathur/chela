use chela::*;

#[test]
fn test_autograd1() {
    let mut a = Tensor::scalar(2.0f32);
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

#[test]
fn test_autograd2() {
    let mut a = Tensor::scalar(2.0f32);
    let mut b = Tensor::scalar(3.0);
    let mut c = Tensor::scalar(5.0);

    a.set_requires_grad(true);
    b.set_requires_grad(true);
    c.set_requires_grad(true);

    let d = &c + &a;
    let e = &b * &d;
    e.backward(1.0);
    
    // e = b(c + a) = bc + ba
    // de/da = b
    // de/db = c + a
    // de/dc = b

    assert_eq!(a.gradient().unwrap(), b);
    assert_eq!(b.gradient().unwrap(), &c + &a);
    assert_eq!(c.gradient().unwrap(), b);
}

#[test]
fn test_autograd3() {
    let mut a = Tensor::scalar(2.0f64);
    let mut b = Tensor::scalar(3.0);
    let mut c = Tensor::scalar(5.0);

    a.set_requires_grad(true);
    b.set_requires_grad(true);
    c.set_requires_grad(true);

    let d = &c + &a;  // 7
    let e = &d * &b;  // 21
    let f = &e + &a;  // 23
    let g = &f * &d;  // 161
    g.backward(1.0);
    
    // g = fd = bc^2 + 2abc + ba^2 + ac + a^2

    // dg/da = 2bc + 2ab + 2a + c
    // dg/db = a^2 + c^2 + 2ac
    // dg/dc = 2ab + 2bc + a

    assert_eq!(a.gradient().unwrap(), (&b * &c * 2.0) + (&b * &a * 2.0) + (&a * 2.0) + &c);
    assert_eq!(b.gradient().unwrap(), (&a * &a) + (&c * &c) + (&a * &c * 2.0));
    assert_eq!(c.gradient().unwrap(), (&a * &b * 2.0) + (&b * &c * 2.0) + &a);
}
