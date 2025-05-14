use chela::*;
use std::env;

use cpu_time::ProcessTime;


fn profile(_: usize) -> u128 {
    let i = 10;
    let j = 100;
    let k = 10000;

    let tensor_a: Tensor<f32> = Tensor::rand([i, k]);
    let tensor_b: Tensor<f32> = Tensor::rand([j, k]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ik", "jk"], "ij"));
    start.elapsed().as_nanos()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let size = args[1].parse::<usize>().unwrap();

    println!("{}", profile(size));
}
