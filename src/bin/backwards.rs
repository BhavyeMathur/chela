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

fn main() {
    let args: Vec<String> = env::args().collect();
    let id = args[1].parse::<usize>().unwrap();

    let time =
        if id == 0 { backward0() }

        else { panic!("invalid ID") };

    println!("{}", time);
}
