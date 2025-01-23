use chela::*;
use std::env;

use cpu_time::ProcessTime;


fn profile(size: usize) -> u128 {
    let mut tensor = Tensor::rand(size);

    let start = ProcessTime::now();
    tensor.sum_along(Axis(0));
    start.elapsed().as_nanos()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let size = args[1].parse::<usize>().unwrap();

    println!("{}", profile(size));
}
