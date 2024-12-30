use chela::*;
use std::env;
use std::time::Instant;


fn profile(size: usize) -> u128 {
    let mut tensor = Tensor::zeros(size);

    let now = Instant::now();
    tensor.fill(5_f32);
    now.elapsed().as_nanos()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let size = args[1].parse::<usize>().unwrap();

    println!("{}", profile(size));
}
