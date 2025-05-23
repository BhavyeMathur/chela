use chela::*;
use std::env;

use cpu_time::ProcessTime;


fn profile(size: usize) -> u128 {
    let tensor: Tensor<f32> = Tensor::rand([size, 2]);

    let start = ProcessTime::now();
    tensor.sum_along(0);
    start.elapsed().as_nanos()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let size = args[1].parse::<usize>().unwrap();

    println!("{}", profile(size));
}
