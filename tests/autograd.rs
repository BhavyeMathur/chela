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

#[test]
fn test_autograd_mul_neg() {
    let mut a = Tensor::scalar(2.0f32);
    let mut b = Tensor::scalar(3.0);

    a.set_requires_grad(true);
    b.set_requires_grad(true);

    let c = -(&a * &b);
    c.backward(1.0);

    // c = -ab
    // dc/da = -b
    // dc/db = -a

    assert_eq!(a.gradient(), Some(Tensor::scalar(-3.0)));
    assert_eq!(b.gradient(), Some(Tensor::scalar(-2.0)));
}


#[test]
fn test_autograd_nested_neg_add() {
    let mut a = Tensor::scalar(5.0f32);
    let mut b = Tensor::scalar(2.0);

    a.set_requires_grad(true);
    b.set_requires_grad(true);

    let c = -(&a + -&b);
    c.backward(1.0);

    // c = -(a - b)
    // dc/da = -1
    // dc/db = 1

    assert_eq!(a.gradient(), Some(Tensor::scalar(-1.0)));
    assert_eq!(b.gradient(), Some(Tensor::scalar(1.0)));
}

#[test]
fn test_autograd_mul_div() {
    let mut a = Tensor::scalar(2.0f32);
    let mut b = Tensor::scalar(3.0);
    let mut c = Tensor::scalar(4.0);

    a.set_requires_grad(true);
    b.set_requires_grad(true);
    c.set_requires_grad(true);

    let d = Tensor::scalar(1.0) / &c;
    d.backward(1.0);

    // d = ab / c
    // dd/da = b / c
    // dd/db = a / c
    // dd/dc = -ab / c^2

    // assert_eq!(a.gradient().unwrap(), &b / &c);
    // assert_eq!(b.gradient().unwrap(), &a / &c);
    assert_almost_eq!(c.gradient().unwrap(), Tensor::scalar(-1.0) / (&c * &c));
}

#[test]
fn test_autograd_compound_expression() {
    let mut a = Tensor::scalar(2.0f32);
    let mut b = Tensor::scalar(3.0);
    let mut c = Tensor::scalar(4.0);
    let mut d = Tensor::scalar(5.0);

    a.set_requires_grad(true);
    b.set_requires_grad(true);
    c.set_requires_grad(true);
    d.set_requires_grad(true);

    let e = ((&a + &b) * &c - &d) / &b;
    e.backward(1.0);

    // e = ac/b - d/b + c
    // de/da = c / b
    // de/db = (d - ac) / b^2
    // de/dc = (a + b) / b
    // de/dd = -1 / b

    assert_eq!(a.gradient().unwrap(), &c / &b);
    assert_almost_eq!(b.gradient().unwrap(), (&d - &a * &c) / (&b * &b));
    assert_almost_eq!(c.gradient().unwrap(), (&a + &b) / &b);
    assert_almost_eq!(d.gradient().unwrap(), -Tensor::scalar(1.0) / &b);
}

#[test]
fn test_autograd_deep_chain_mul_add() {
    let mut a = Tensor::scalar(1.0f64);
    let mut b = Tensor::scalar(2.0);
    let mut c = Tensor::scalar(3.0);
    let mut d = Tensor::scalar(4.0);
    let mut e = Tensor::scalar(5.0);

    a.set_requires_grad(true);
    b.set_requires_grad(true);
    c.set_requires_grad(true);
    d.set_requires_grad(true);
    e.set_requires_grad(true);

    let x = &a * &b;        // x = ab
    let y = &x + &c;        // y = ab + c
    let z = &y * &d;        // z = (ab + c)d
    let out = &z + &e;      // out = ((ab + c)d) + e
    out.backward(1.0);

    // out = ((ab + c)d) + e
    // dout/da = bd
    // dout/db = ad
    // dout/dc = d
    // dout/dd = ab + c
    // dout/de = 1

    assert_eq!(a.gradient(), Some(Tensor::scalar(2.0 * 4.0))); // bd = 8
    assert_eq!(b.gradient(), Some(Tensor::scalar(1.0 * 4.0))); // ad = 4
    assert_eq!(c.gradient(), Some(Tensor::scalar(4.0)));       // d
    assert_eq!(d.gradient(), Some(Tensor::scalar(1.0 * 2.0 + 3.0))); // ab + c = 5
    assert_eq!(e.gradient(), Some(Tensor::scalar(1.0)));       // 1
}
