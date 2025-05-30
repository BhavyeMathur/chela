use chela::*;
use std::env;
use cpu_time::ProcessTime;

fn profile(size: usize) -> u128 {
    let start = ProcessTime::now();
    let _tensor: NdArray<f32> = NdArray::ones(size);
    start.elapsed().as_nanos()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let size = args[1].parse::<usize>().unwrap();

   println!("{}", profile(size));
}
