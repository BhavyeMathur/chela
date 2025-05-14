use chela::*;

use cpu_time::ProcessTime;


fn profile() -> u128 {
    let i = 10;
    let j = 100;
    let k = 10000;

    let tensor_a: Tensor<f32> = Tensor::rand([i, k]);
    let tensor_b: Tensor<f32> = Tensor::rand([j, k]);

    let start = ProcessTime::now();
    _ = einsum(&[&tensor_a, &tensor_b], (["ik", "jk"], "ij"));
    start.elapsed().as_nanos()
}

fn main() {
    println!("{}", profile());
}
