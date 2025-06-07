use redstone_ml::*;
use std::env;

use redstone_ml::profiler::profile_func;

type T = f32;
const M: usize = 100;


fn backward0() {
    let n = 1000;

    let mut tensor_a = Tensor::<T>::rand([n]);
    let mut tensor_b = Tensor::<T>::rand([n]);
    let mut tensor_c = Tensor::<T>::rand([n]);

    tensor_a.set_requires_grad(true);
    tensor_b.set_requires_grad(true);
    tensor_c.set_requires_grad(true);

    let ones = NdArray::<T>::ones([n]);

    let func = || {
        for _ in 0..M {
            let result = (&tensor_a * &tensor_b) / (&tensor_c + 1.0);
            result.backward_with(&ones);

            tensor_a.zero_gradient();
            tensor_b.zero_gradient();
            tensor_c.zero_gradient();
        }
    };
    profile_func(func)
}

fn backward1() {
    let i = 1000;
    let j = 500;

    let x = Tensor::<T>::rand([j]);
    let mut a = Tensor::<T>::rand([i, j]);
    let mut b = Tensor::<T>::rand([i]);

    a.set_requires_grad(true);
    b.set_requires_grad(true);

    let ones = NdArray::<T>::ones([i]);

    let func = || {
        for _ in 0..M {
            let result = a.matmul(&x) + &b;
            result.backward_with(&ones);

            a.zero_gradient();
            b.zero_gradient();
        }
    };
    profile_func(func)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let test_id = args[1].parse::<usize>().unwrap();

    match test_id {
        0 => { backward0() },
        1 => { backward1() },
        _ => { panic!("invalid ID") },
    }
}
