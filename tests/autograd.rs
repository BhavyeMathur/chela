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

#[test]
fn test_autograd2() {
    let mut a = Tensor::scalar(2.0);
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
    
    // dg/df = d = 7
    // dg/dd = f = 23
    
    // df/de = 1
    // df/da = 1
    // dg/de = (dg/df)(df/de) = 7
    // dg/da = (dg/df)(df/de) = 7

    // g = (b(c + a) + a)(c + a)
    // dg/da = 2 * b(c + a) + c + 2a
    // dg/db = (c + a)^2
    // dg/dc = b(c + a) + a

    assert_eq!(a.gradient(), Some((&d * &b * 2.0) + &d + &a));
    assert_eq!(b.gradient(), Some((&c + &a) * (&c + &a)));
    assert_eq!(c.gradient(), Some((&c + &a) * &b + &a));
}
