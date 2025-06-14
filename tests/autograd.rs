use redstone_ml::*;

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
    let mut a = Tensor::new([2.0f64, 4.0, 6.0]);
    let mut b = Tensor::scalar(3.0);
    let mut c = Tensor::new([[2.0, 4.0, 6.0], [5.0, 7.0, 11.0]]);

    a.set_requires_grad(true);
    b.set_requires_grad(true);
    c.set_requires_grad(true);

    let d = &c + &a;
    d.backward();

    assert_eq!(a.gradient().unwrap(), Tensor::new([2.0, 2.0, 2.0]));
    assert_eq!(b.gradient().unwrap(), Tensor::scalar(0.0));
    assert_eq!(c.gradient().unwrap(), Tensor::<f64>::ones([2, 3]));

    let e = &d * &b;
    a.zero_gradient();
    b.zero_gradient();
    c.zero_gradient();
    e.backward();

    assert_eq!(a.gradient().unwrap(), Tensor::new([6.0, 6.0, 6.0]));
    assert_eq!(b.gradient().unwrap(), Tensor::scalar(59.0));
    assert_eq!(c.gradient().unwrap(), Tensor::<f64>::ones([2, 3]) * 3.0);

    let f = &e + &a;
    a.zero_gradient();
    b.zero_gradient();
    c.zero_gradient();
    f.backward();

    assert_eq!(a.gradient().unwrap(), Tensor::new([8.0, 8.0, 8.0]));
    assert_eq!(b.gradient().unwrap(), Tensor::scalar(59.0));
    assert_eq!(c.gradient().unwrap(), Tensor::<f64>::ones([2, 3]) * 3.0);

    let g = &f * &d;
    a.zero_gradient();
    b.zero_gradient();
    c.zero_gradient();
    g.backward();

    assert_eq!(a.gradient().unwrap(), Tensor::new([81.0, 141.0, 215.0]));
    assert_eq!(b.gradient().unwrap(), Tensor::scalar(683.0));
    assert_eq!(c.gradient().unwrap(), Tensor::new([[26.0, 52.0, 78.0], [44.0, 70.0, 108.0]]));
}

#[test]
fn test_autograd5() {
    let mut a = Tensor::new([1.0f32, 2.0, 3.0]);  // [3]
    let mut b = Tensor::new([[2.0, -2.0, 1.0], [-1.0, -2.5, 2.0], [-3.0, 3.0, 2.5]]);  // [3, 3]
    let mut c = Tensor::new([[2.0], [4.0], [6.0]]);  // [3, 1]

    a.set_requires_grad(true);
    b.set_requires_grad(true);
    c.set_requires_grad(true);

    let x = (&a / &b) + &c;
    let y = (&c - &b) / (&c + &a);
    let z = -&x + &y - (&x * &x) + (&x * &y) / &a;
    z.backward();

    assert_almost_eq!(a.gradient().unwrap(), NdArray::new([-5.6277f32, -3.5112, -23.9599]), 1e-4);

    assert_almost_eq!(b.gradient().unwrap(), NdArray::new([[ 0.3333, 0.8750, 32.2667],
                                                           [ 5.2f32, 1.7613, 8.5238],
                                                           [ 0.2751, 2.6019, 6.9520]]), 1e-4);

    assert_almost_eq!(c.gradient().unwrap(), NdArray::new([[-17.84f32], [-24.5101], [-40.1665]]), 1e-4);
}

#[test]
fn test_autograd6() {
    let mut a = Tensor::new([1.0f32, 2.0, 3.0]);  // [3]
    let mut b = Tensor::new([[2.0, -2.0, 1.0], [-1.0, -2.5, 2.0], [-3.0, 3.0, 2.5]]);  // [3, 3]
    let mut c = Tensor::new([[2.0], [4.0], [6.0]]);  // [3, 1]

    a.set_requires_grad(true);
    b.set_requires_grad(true);
    c.set_requires_grad(true);

    let x = (&a / 2.0) + &c;
    let y = (&c - &b - 2.0) / (&c + 6.0);
    let z = -&x + &y - (&x * 5.0) + (&x * &y) / &a;
    z.backward();

    assert_almost_eq!(a.gradient().unwrap(), NdArray::new([-13.2, -9.7, -9.0556]), 1e-4);

    assert_almost_eq!(b.gradient().unwrap(), NdArray::new([[-0.4375, -0.3125, -0.2708],
                                                           [ -0.55, -0.35, -0.2833],
                                                           [ -0.6250, -0.375, -0.2917]]), 1e-4);

    assert_almost_eq!(c.gradient().unwrap(), NdArray::new([[-17.0807], [-16.6142], [-16.4740]]), 1e-4);
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

    let expected = (&c / &b).into_ndarray();
    assert_almost_eq!(a.gradient().unwrap(), expected);

    let expected = ((&d - &a * &c) / (&b * &b)).into_ndarray();
    assert_almost_eq!(b.gradient().unwrap(), expected);

    let expected = ((&a + &b) / &b).into_ndarray();
    assert_almost_eq!(c.gradient().unwrap(), expected);

    let expected = -NdArray::scalar(1.0) / b.into_ndarray();
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
fn test_autograd_matmul_ops() {
    let mut vector1 = Tensor::new([1.0, 2.0, 3.0]);
    let mut vector2 = Tensor::new([-1.0, 5.0, -9.0]);

    let mut matrix1 = Tensor::new([[2.0, -2.0, 1.0], [-1.0, -2.5, 2.0], [-3.0, 3.0, 2.5]]);
    let mut matrix2 = Tensor::new([[1.0, -5.0, 4.0], [-3.0, -5.5, 3.0], [-9.0, 2.0, 1.5]]);

    vector1.set_requires_grad(true);
    vector2.set_requires_grad(true);
    matrix1.set_requires_grad(true);
    matrix2.set_requires_grad(true);


    // Dot Product

    let x = vector1.dot(&vector2);
    x.backward();

    assert_eq!(vector1.gradient().unwrap(), vector2.ndarray());
    assert_eq!(vector2.gradient().unwrap(), vector1.ndarray());

    let x = vector1.matmul(&vector2);
    vector1.zero_gradient();
    vector2.zero_gradient();
    vector2.set_requires_grad(false);
    x.backward();

    assert_eq!(vector1.gradient().unwrap(), vector2.ndarray());
    assert_eq!(vector2.gradient(), None);

    vector2.set_requires_grad(true);

    // Matrix-Vector Product 1

    let x = matrix1.matmul(&vector1);
    vector1.zero_gradient();
    x.backward();

    assert_eq!(matrix1.gradient().unwrap(), NdArray::new([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]));
    assert_eq!(vector1.gradient().unwrap(), NdArray::new([-2.0, -1.5, 5.5]));


    // Matrix-Vector Product 1

    let x = matrix1.matmul(&vector2);
    matrix1.zero_gradient();
    vector2.zero_gradient();
    x.backward();

    assert_eq!(matrix1.gradient().unwrap(), NdArray::new([[-1.0, 5.0, -9.0], [-1.0, 5.0, -9.0], [-1.0, 5.0, -9.0]]));
    assert_eq!(vector2.gradient().unwrap(), NdArray::new([-2.0, -1.5, 5.5]));


    // Matrix-Matrix Product

    let x = matrix1.matmul(&matrix2);
    matrix1.zero_gradient();
    x.backward();

    assert_eq!(matrix1.gradient().unwrap(), NdArray::new([[0.0, -5.5, -5.5], [0.0, -5.5, -5.5], [0.0, -5.5, -5.5]]));
    assert_eq!(matrix2.gradient().unwrap(), NdArray::new([[-2.0, -2.0, -2.0], [-1.5, -1.5, -1.5], [5.5, 5.5, 5.5]]));
}


#[test]
fn test_autograd_bmm_ops() {
    let mut matrix1 = Tensor::new([
        [[2.0, -2.0, 1.0], [-1.0, -2.5, 2.0], [-3.0, 3.0, 2.5]],
        [[-3.0, 3.0, 8.0], [6.0, -3.0, -1.0], [5.0, 1.5, -4.0]],
        [[4.0, 4.0, 7.0], [-4.0, 2.5, 6.5], [-8.0, 4.0, 9.0]]]);

    let mut matrix2 = Tensor::new([
        [[1.0, -5.0, 4.0], [-3.0, -5.5, 3.0], [-9.0, 2.0, 1.5]],
        [[4.0, 9.0, 5.5], [8.5, -1.5, 3.5], [3.0, -7.0, 2.0]],
        [[2.0, 1.0, -1.0], [0.0, 6.5, -9.5], [-7.0, -1.0, 4.5]]]);

    matrix1.set_requires_grad(true);
    matrix2.set_requires_grad(true);


    // Batch Matrix Multiplication

    let x = matrix1.bmm(&matrix2);
    x.backward();

    assert_eq!(matrix1.gradient().unwrap(),
               NdArray::new([
                   [[0.0, -5.50, -5.50], [0.0, -5.50, -5.50], [0.0, -5.50, -5.50]],
                   [[18.50, 10.50, -2.0], [18.50, 10.50, -2.0], [18.50, 10.50, -2.0]],
                   [[2.0, -3.0, -3.50], [2.0, -3.0, -3.50], [2.0, -3.0, -3.50]]]));

    assert_eq!(matrix2.gradient().unwrap(),
               NdArray::new([
                   [[-2.0, -2.0, -2.0], [-1.50, -1.50, -1.50], [5.50, 5.50, 5.50]],
                   [[8.0, 8.0, 8.0], [1.50, 1.50, 1.50], [3.0, 3.0, 3.0]],
                   [[-8.0, -8.0, -8.0], [10.50, 10.50, 10.50], [22.50, 22.50, 22.50]]]));
}

#[test]
fn test_autograd_transpose_ops() {
    let mut matrix1 = Tensor::new([
        [[2.0, -2.0], [-1.0, -2.5], [-3.0, 3.0]],
        [[-3.0, 3.0], [6.0, -3.0], [5.0, 1.5]],
        [[4.0, 4.0], [-4.0, 2.5], [-8.0, 4.0]],
        [[4.0, 4.0], [-4.0, 2.5], [-8.0, 4.0]]]);  // [4, 3, 2]

    let mut matrix2 = Tensor::new([
        [[1.0, -5.0], [-3.0, -5.5], [-9.0, 2.0]],
        [[4.0, 9.0], [8.5, -1.5], [3.0, -7.0]],
        [[2.0, 1.0], [0.0, 6.5], [-7.0, -1.0]],
        [[2.0, 1.0], [0.0, 6.5], [-7.0, -1.0]]]);  // [4, 3, 2] 

    matrix1.set_requires_grad(true);
    matrix2.set_requires_grad(true);

    let mat1 = (&matrix1).view();
    let mat2 = (&matrix2).view();

    let matrix3 = (&matrix1).transpose(1, 2);

    let x = matrix3.bmm(&matrix2);
    x.backward();

    let matrix1_correct = NdArray::new([
        [[-4.000, -4.000], [-8.500, -8.500], [-7.0000, -7.0000]],
        [[13.000, 13.000], [7.0000, 7.0000], [-4.0000, -4.0000]],
        [[3.0000, 3.0000], [6.5000, 6.5000], [-8.0000, -8.0000]],
        [[3.0000, 3.0000], [6.5000, 6.5000], [-8.0000, -8.0000]]]);

    let matrix2_correct = NdArray::new([
        [[0.0000, 0.0000], [-3.500, -3.500], [0.0000, 0.0000]],
        [[0.0000, 0.0000], [3.0000, 3.0000], [6.5000, 6.5000]],
        [[8.0000, 8.0000], [-1.500, -1.500], [-4.00, -4.0000]],
        [[8.0000, 8.0000], [-1.500, -1.500], [-4.00, -4.0000]]]);

    assert_eq!(matrix1.gradient().unwrap(), matrix1_correct);
    assert_eq!(matrix2.gradient().unwrap(), matrix2_correct);

    let matrix3 = mat1.transpose(1, 2);
    let x = matrix3.bmm(mat2);
    matrix1.zero_gradient();
    matrix2.zero_gradient();
    x.backward();
    
    assert_eq!(matrix1.gradient().unwrap(), matrix1_correct);
    assert_eq!(matrix2.gradient().unwrap(), matrix2_correct);
}

#[test]
fn test_autograd_reshape() {
    let mut vector1 = Tensor::new([1.0, 2.0, 3.0, 4.0]);
    let mut vector2 = Tensor::new([-1.0, 5.0, -9.0, 12.0]);
    vector1.set_requires_grad(true);
    vector2.set_requires_grad(true);

    let mat1 = (&vector1).reshape([2, 2]);
    let mat2 = (&vector2).reshape([2, 2]);
    let result = mat1.matmul(&mat2);

    vector1.zero_gradient();
    vector2.zero_gradient();
    result.backward();

    assert_eq!(vector1.gradient().unwrap(), NdArray::new([4.0, 3.0, 4.0, 3.0]));
    assert_eq!(vector2.gradient().unwrap(), NdArray::new([4.0, 4.0, 6.0, 6.0]));
}

fn test_autograd_view() {
    let mut vector1 = Tensor::new([1.0, 2.0, 3.0, 4.0]);
    let mut vector2 = Tensor::new([-1.0, 5.0, -9.0, 12.0]);
    vector1.set_requires_grad(true);
    vector2.set_requires_grad(true);

    let vector_view = (&vector1).view();

    let result = vector_view.dot(&vector2);
    result.backward();

    assert_eq!(vector1.gradient().unwrap(), vector2.ndarray());
    assert_eq!(vector_view.gradient(), None);
}

fn test_autograd_squeeze() {
    let mut vector1 = Tensor::new([[[1.0], [2.0], [3.0], [4.0]]]);
    let mut vector2 = Tensor::new([[-1.0], [5.0], [-9.0], [12.0]]);
    vector1.set_requires_grad(true);
    vector2.set_requires_grad(true);

    let vec1 = (&vector1).squeeze();
    let vec2 = (&vector2).squeeze();

    let mat1 = (&vec1).reshape([2, 2]);
    let mat2 = (&vec2).reshape([2, 2]);
    let result = mat1.matmul(&mat2);

    result.backward();
    assert_eq!(vector1.gradient().unwrap(), NdArray::new([[[4.0], [3.0], [4.0], [3.0]]]));
    assert_eq!(vector2.gradient().unwrap(), NdArray::new([[4.0], [4.0], [6.0], [6.0]]));
}

fn test_autograd_moved_reshape() {
    let mut vector1 = Tensor::new([1.0, 2.0, 3.0, 4.0]);
    let mut vector2 = Tensor::new([-1.0, 5.0, -9.0, 12.0]);

    vector1.set_requires_grad(true);
    vector2.set_requires_grad(true);

    let mat1 = (vector1 + &vector2).reshape([2, 2]);
    let mat2 = vector2.reshape([2, 2]);

    let result = mat1.matmul(&mat2);

    result.backward();
    assert_eq!(mat1.gradient().unwrap(), NdArray::new([[4.0, 3.0], [4.0, 3.0]]));
    assert_eq!(mat2.gradient().unwrap(), NdArray::new([[-2.0, -3.0], [27.0, 26.0]]));
}
