use chela::*;

use cpu_time::ProcessTime;


fn profile() -> u128 {
    let a = 100;
    let b = 5;
    let c = 20;
    let d = 50;
    let e = 100;

    let tensor_a: Tensor<f32> = Tensor::rand([a, b, c]);
    let tensor_b: Tensor<f32> = Tensor::rand([c, d]);
    let tensor_c: Tensor<f32> = Tensor::rand([d, e]);

    let start = ProcessTime::now();
    _ = einsum(&[&tensor_a, &tensor_b, &tensor_c], (["abc", "cd", "de"], "ae"));
    start.elapsed().as_nanos()
}

fn main() {
    println!("{}", profile());
}
