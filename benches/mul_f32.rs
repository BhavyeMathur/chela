use chela::*;
use std::env;

use cpu_time::ProcessTime;

fn profile(size: usize) -> u128 {
    let tensor1: Tensor<f32> = Tensor::rand(size);
    let tensor2: Tensor<f32> = Tensor::rand(size);

    let start = ProcessTime::now();
    let _ = tensor1 * tensor2;
    start.elapsed().as_nanos()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let size = args[1].parse::<usize>().unwrap();

    println!("{}", profile(size));
}
