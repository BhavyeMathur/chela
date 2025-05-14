use chela::*;

use cpu_time::ProcessTime;


fn profile() -> u128 {
    let i = 10;
    let j = 100;
    let k = 1000;
    let m = 50;

    let tensor_a: Tensor<f32> = Tensor::rand([i, j]);
    let tensor_b: Tensor<f32> = Tensor::rand([k, m]);

    let start = ProcessTime::now();
    _ = einsum(&[&tensor_a, &tensor_b], (["ij", "km"], "im"));
    start.elapsed().as_nanos()
}

fn main() {
    println!("{}", profile());
}
