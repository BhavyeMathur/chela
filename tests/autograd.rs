use chela::*;

#[test]
fn test_autograd1() {
    let mut a = Tensor::scalar(2.0f32);
    let mut b = Tensor::scalar(3.0);

    a.set_requires_grad(true);
    b.set_requires_grad(true);

    let c = &a * &b;
    let d = &c * &b;
    d.backward();

    // d = ab^2
    // dd/da = b^2
    // dd/db = 2ab

    assert_eq!(a.gradient().unwrap(), &b * &b);
    assert_eq!(b.gradient().unwrap(), &a * &b * 2.0);
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
    e.backward();

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
    g.backward();

    // g = fd = bc^2 + 2abc + ba^2 + ac + a^2

    // dg/da = 2bc + 2ab + 2a + c
    // dg/db = a^2 + c^2 + 2ac
    // dg/dc = 2ab + 2bc + a

    assert_eq!(a.gradient().unwrap(), (&b * &c * 2.0) + (&b * &a * 2.0) + (&a * 2.0) + &c);
    assert_eq!(b.gradient().unwrap(), (&a * &a) + (&c * &c) + (&a * &c * 2.0));
    assert_eq!(c.gradient().unwrap(), (&a * &b * 2.0) + (&b * &c * 2.0) + &a);
}

#[test]
fn test_autograd4() {
    let mut a = Tensor::from([2.0f64, 4.0, 6.0]);
    let mut b = Tensor::scalar(3.0);
    let mut c = Tensor::from([[2.0, 4.0, 6.0], [5.0, 7.0, 11.0]]);

    a.set_requires_grad(true);
    b.set_requires_grad(true);
    c.set_requires_grad(true);

    let d = &c + &a;
    d.backward();

    assert_eq!(a.gradient().unwrap(), Tensor::from([2.0, 2.0, 2.0]));
    assert_eq!(b.gradient().unwrap(), Tensor::scalar(0.0));
    assert_eq!(c.gradient().unwrap(), Tensor::<f64>::ones([2, 3]));

    let e = &d * &b;
    a.zero_gradient();
    b.zero_gradient();
    c.zero_gradient();
    e.backward();

    assert_eq!(a.gradient().unwrap(), Tensor::from([6.0, 6.0, 6.0]));
    assert_eq!(b.gradient().unwrap(), Tensor::scalar(59.0));
    assert_eq!(c.gradient().unwrap(), Tensor::<f64>::ones([2, 3]) * 3.0);

    let f = &e + &a;
    a.zero_gradient();
    b.zero_gradient();
    c.zero_gradient();
    f.backward();

    assert_eq!(a.gradient().unwrap(), Tensor::from([8.0, 8.0, 8.0]));
    assert_eq!(b.gradient().unwrap(), Tensor::scalar(59.0));
    assert_eq!(c.gradient().unwrap(), Tensor::<f64>::ones([2, 3]) * 3.0);

    let g = &f * &d;
    a.zero_gradient();
    b.zero_gradient();
    c.zero_gradient();
    g.backward();

    assert_eq!(a.gradient().unwrap(), Tensor::from([81.0, 141.0, 215.0]));
    assert_eq!(b.gradient().unwrap(), Tensor::scalar(683.0));
    assert_eq!(c.gradient().unwrap(), Tensor::from([[26.0, 52.0, 78.0], [44.0, 70.0, 108.0]]));
}

#[test]
fn test_autograd5() {
    let mut a = Tensor::from([1.0f32, 2.0, 3.0]);  // [3]
    let mut b = Tensor::from([[2.0, -2.0, 1.0], [-1.0, -2.5, 2.0], [-3.0, 3.0, 2.5]]);  // [3, 3]
    let mut c = Tensor::from([[2.0], [4.0], [6.0]]);  // [3, 1]

    a.set_requires_grad(true);
    b.set_requires_grad(true);
    c.set_requires_grad(true);

    let x = (&a / &b) + &c;
    let y = (&c - &b) / (&c + &a);
    let z = -&x + &y - (&x * &x) + (&x * &y) / &a;
    z.backward();

    assert_almost_eq!(a.gradient().unwrap(), NdArray::from([-5.6277f32, -3.5112, -23.9599]), 1e-4);

    assert_almost_eq!(b.gradient().unwrap(), NdArray::from([[ 0.3333, 0.8750, 32.2667],
                                                           [ 5.2f32, 1.7613, 8.5238],
                                                           [ 0.2751, 2.6019, 6.9520]]), 1e-4);

    assert_almost_eq!(c.gradient().unwrap(), NdArray::from([[-17.84f32], [-24.5101], [-40.1665]]), 1e-4);
}

#[test]
fn test_autograd6() {
    let mut a = Tensor::from([1.0f32, 2.0, 3.0]);  // [3]
    let mut b = Tensor::from([[2.0, -2.0, 1.0], [-1.0, -2.5, 2.0], [-3.0, 3.0, 2.5]]);  // [3, 3]
    let mut c = Tensor::from([[2.0], [4.0], [6.0]]);  // [3, 1]

    a.set_requires_grad(true);
    b.set_requires_grad(true);
    c.set_requires_grad(true);

    let x = (&a / 2.0) + &c;
    let y = (&c - &b - 2.0) / (&c + 6.0);
    let z = -&x + &y - (&x * 5.0) + (&x * &y) / &a;
    z.backward();

    assert_almost_eq!(a.gradient().unwrap(), NdArray::from([-13.2, -9.7, -9.0556]), 1e-4);

    assert_almost_eq!(b.gradient().unwrap(), NdArray::from([[-0.4375, -0.3125, -0.2708],
                                                           [ -0.55, -0.35, -0.2833],
                                                           [ -0.6250, -0.375, -0.2917]]), 1e-4);

    assert_almost_eq!(c.gradient().unwrap(), NdArray::from([[-17.0807], [-16.6142], [-16.4740]]), 1e-4);
}

#[test]
fn test_autograd_mul_neg() {
    let mut a = Tensor::scalar(2.0f32);
    let mut b = Tensor::scalar(3.0);

    a.set_requires_grad(true);
    b.set_requires_grad(true);

    let c = -(&a * &b);
    c.backward();

    // c = -ab
    // dc/da = -b
    // dc/db = -a

    assert_eq!(a.gradient().unwrap(), -b.ndarray());
    assert_eq!(b.gradient(), Some(NdArray::scalar(-2.0)));
}


#[test]
fn test_autograd_nested_neg_add() {
    let mut a = Tensor::scalar(5.0f32);
    let mut b = Tensor::scalar(2.0);

    a.set_requires_grad(true);
    b.set_requires_grad(true);

    let c = -(&a + -&b);
    c.backward();

    // c = -(a - b)
    // dc/da = -1
    // dc/db = 1

    assert_eq!(a.gradient().unwrap(), NdArray::scalar(-1.0));
    assert_eq!(b.gradient().unwrap(), NdArray::scalar(1.0));
}

#[test]
fn test_autograd_mul_div() {
    let mut a = Tensor::scalar(2.0f32);
    let mut b = Tensor::scalar(3.0);
    let mut c = Tensor::scalar(4.0);

    a.set_requires_grad(true);
    b.set_requires_grad(true);
    c.set_requires_grad(true);

    let d = &a * &b / &c;
    d.backward();

    // d = ab / c
    // dd/da = b / c
    // dd/db = a / c
    // dd/dc = -ab / c^2

    assert_eq!(a.gradient().unwrap(), &b / &c);
    assert_eq!(b.gradient().unwrap(), &a / &c);
    assert_almost_eq!(c.gradient().unwrap(), (-&a * &b / (&c * &c)).into_ndarray());
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
    e.backward();

    // e = ac/b - d/b + c
    // de/da = c / b
    // de/db = (d - ac) / b^2
    // de/dc = (a + b) / b
    // de/dd = -1 / b

    let expected =  (&c / &b).into_ndarray();
    assert_almost_eq!(a.gradient().unwrap(), expected);

    let expected = ((&d - &a * &c) / (&b * &b)).into_ndarray();
    assert_almost_eq!(b.gradient().unwrap(), expected);

    let expected =  ((&a + &b) / &b).into_ndarray();
    assert_almost_eq!(c.gradient().unwrap(), expected);

    let expected =  -NdArray::scalar(1.0) / b.into_ndarray();
    assert_almost_eq!(d.gradient().unwrap(), expected);
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
    out.backward();

    // out = ((ab + c)d) + e
    // dout/da = bd
    // dout/db = ad
    // dout/dc = d
    // dout/dd = ab + c
    // dout/de = 1

    assert_eq!(a.gradient(), Some(NdArray::scalar(2.0 * 4.0))); // bd = 8
    assert_eq!(b.gradient(), Some(NdArray::scalar(1.0 * 4.0))); // ad = 4
    assert_eq!(c.gradient(), Some(NdArray::scalar(4.0)));       // d
    assert_eq!(d.gradient(), Some(NdArray::scalar(1.0 * 2.0 + 3.0))); // ab + c = 5
    assert_eq!(e.gradient(), Some(NdArray::scalar(1.0)));       // 1
}

#[test]
fn test_autograd_matrix_vector_ops() {
    let mut vector1 = Tensor::from([1.0, 2.0, 3.0]);
    let mut vector2 = Tensor::from([-1.0, 5.0, -9.0]);
    
    let mut matrix = Tensor::from([[2.0, -2.0, 1.0], [-1.0, -2.5, 2.0], [-3.0, 3.0, 2.5]]);

    vector1.set_requires_grad(true);
    vector2.set_requires_grad(true);
    matrix.set_requires_grad(true);
    
    let x = vector1.dot(&vector2);
    x.backward();
    
    assert_eq!(vector1.gradient().unwrap(), vector2.ndarray());
    assert_eq!(vector2.gradient().unwrap(), vector1.ndarray());
}
