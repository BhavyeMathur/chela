use chela::*;
use std::env;

use cpu_time::ProcessTime;


type T = f32;


fn backward0() -> u128 {
    let n = 1000;

    let mut tensor_a = Tensor::<T>::rand([n]);
    let mut tensor_b = Tensor::<T>::rand([n]);
    let mut tensor_c = Tensor::<T>::rand([n]);
    
    tensor_a.set_requires_grad(true);
    tensor_b.set_requires_grad(true);
    tensor_c.set_requires_grad(true);

    let ones = NdArray::<T>::ones([n]);

    let start = ProcessTime::now();

    for _ in 0..1000 {
        let result = (&tensor_a * &tensor_b) / (&tensor_c + 1.0);
        result.backward_with(&ones);

        tensor_a.zero_gradient();
        tensor_b.zero_gradient();
        tensor_c.zero_gradient();
    }

    start.elapsed().as_nanos()
}

fn backward1() -> u128 {
    let i = 1000;
    let j = 500;

    let x = Tensor::<T>::rand([j]);
    let mut a = Tensor::<T>::rand([i, j]);
    let mut b = Tensor::<T>::rand([i]);

    a.set_requires_grad(true);
    b.set_requires_grad(true);

    let ones = NdArray::<T>::ones([i]);

    let start = ProcessTime::now();

    for _ in 0..100 {
        let result = a.matmul(&x) + &b;
        result.backward_with(&ones);

        a.zero_gradient();
        b.zero_gradient();
    }

    start.elapsed().as_nanos()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let id = args[1].parse::<usize>().unwrap();

    let time =
        if id == 0 { backward0() }
        else if id == 1 { backward1() }

        else { panic!("invalid ID") };

    println!("{}", time);
}
