use chela::*;
use std::env;

use cpu_time::ProcessTime;


fn profile(size: usize) -> u128 {
    let tensor = Tensor::zeros([size, 2]);
    let mut tensor_slice = tensor.slice(s![.., 0]);

    let start = ProcessTime::now();
    tensor_slice.fill(5_f32);
    start.elapsed().as_nanos()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let size = if args.len() < 2 { 65536 } else { args[1].parse().unwrap() };

    println!("{}", profile(size));
}
