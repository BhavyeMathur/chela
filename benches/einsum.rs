use chela::*;
use std::env;

use cpu_time::ProcessTime;


fn einsum1() -> u128 {
    let i = 10000;

    let tensor_a: Tensor<f32> = Tensor::rand([i]);
    let tensor_b: Tensor<f32> = Tensor::rand([i]);

    let start = ProcessTime::now();
    _ = einsum(&[&tensor_a, &tensor_b], (["i", "i"], ""));
    start.elapsed().as_nanos()
}

fn einsum2() -> u128 {
    let i = 1000;
    let j = 500;

    let tensor_a: Tensor<f32> = Tensor::rand([i, j]);
    let tensor_b: Tensor<f32> = Tensor::rand([j]);

    let start = ProcessTime::now();
    _ = einsum(&[&tensor_a, &tensor_b], (["ij", "j"], "i"));
    start.elapsed().as_nanos()
}

fn einsum3() -> u128 {
    let i = 100;
    let j = 1000;
    let k = 500;

    let tensor_a: Tensor<f32> = Tensor::rand([i, j]);
    let tensor_b: Tensor<f32> = Tensor::rand([j, k]);

    let start = ProcessTime::now();
    _ = einsum(&[&tensor_a, &tensor_b], (["ij", "jk"], "ik"));
    start.elapsed().as_nanos()
}

fn einsum4() -> u128 {
    let i = 100;
    let j = 1000;
    let k = 500;

    let tensor_a: Tensor<f32> = Tensor::rand([i, k]);
    let tensor_b: Tensor<f32> = Tensor::rand([j, k]);

    let start = ProcessTime::now();
    _ = einsum(&[&tensor_a, &tensor_b], (["ik", "jk"], "ij"));
    start.elapsed().as_nanos()
}

fn einsum5() -> u128 {
    let a = 100;
    let b = 5;
    let c = 20;
    let d = 50;
    let e = 100;

    let tensor_a: Tensor<f32> = Tensor::rand([a, b, c]);
    let tensor_b: Tensor<f32> = Tensor::rand([b, d]);
    let tensor_c: Tensor<f32> = Tensor::rand([d, e]);

    let start = ProcessTime::now();
    _ = einsum(&[&tensor_a, &tensor_b, &tensor_c], (["abc", "bd", "de"], "ae"));
    start.elapsed().as_nanos()
}

fn einsum6() -> u128 {
    let a = 100;
    let b = 5;
    let c = 20;
    let d = 50;
    let e = 100;

    let tensor_a: Tensor<f32> = Tensor::rand([a, b, c]);
    let tensor_b: Tensor<f32> = Tensor::rand([b, d]);
    let tensor_c: Tensor<f32> = Tensor::rand([b, c]);
    let tensor_d: Tensor<f32> = Tensor::rand([d, e]);

    let start = ProcessTime::now();
    _ = einsum(&[&tensor_a, &tensor_b, &tensor_c, &tensor_d], (["abc", "bd", "bc", "de"], "ae"));
    start.elapsed().as_nanos()
}

fn einsum7() -> u128 {
    let i = 128;
    let j = 64;
    let k = 32;

    let tensor_a: Tensor<f32> = Tensor::rand([i, j]);
    let tensor_b: Tensor<f32> = Tensor::rand([k, j]);

    let start = ProcessTime::now();
    _ = einsum(&[&tensor_a, &tensor_b], (["ij", "kj"], "ikj"));
    start.elapsed().as_nanos()
}

fn einsum8() -> u128 {
    let a = 128;
    let b = 64;
    let c = 32;

    let tensor_a: Tensor<f32> = Tensor::rand([a, b, c]);

    let start = ProcessTime::now();
    _ = einsum(&[&tensor_a], (["abc"], ""));
    start.elapsed().as_nanos()
}

fn einsum9() -> u128 {
    let i = 4096;

    let tensor_a: Tensor<f32> = Tensor::rand([i, i]);

    let start = ProcessTime::now();
    _ = einsum_view(&tensor_a, ("ii", "i")).unwrap();
    start.elapsed().as_nanos()
}

fn einsum10() -> u128 {
    let a = 10;
    let b = 20;
    let c = 30;
    let d = 40;

    let tensor_a: Tensor<f32> = Tensor::rand([a, b, c, d]);

    let start = ProcessTime::now();
    _ = einsum_view(&tensor_a, ("abcd", "dcba")).unwrap();
    start.elapsed().as_nanos()
}

fn einsum_2operands_0() -> u128 {
    let i = 100;
    let j = 50;
    let k = 100;

    let tensor_a: Tensor<f32> = Tensor::rand([i, j]);
    let tensor_b: Tensor<f32> = Tensor::rand([j, k]);

    let start = ProcessTime::now();
    _ = einsum(&[&tensor_a, &tensor_b], (["ij", "jk"], ""));
    start.elapsed().as_nanos()
}

fn einsum_2operands_1() -> u128 {
    let i = 100;
    let j = 50;
    let k = 100;

    let tensor_a: Tensor<f32> = Tensor::rand([i, j]);
    let tensor_b: Tensor<f32> = Tensor::rand([j, k]);

    let start = ProcessTime::now();
    _ = einsum(&[&tensor_a, &tensor_b], (["ij", "jk"], "i"));
    start.elapsed().as_nanos()
}

fn einsum_2operands_2() -> u128 {
    let i = 100;
    let j = 50;
    let k = 100;

    let tensor_a: Tensor<f32> = Tensor::rand([i, j]);
    let tensor_b: Tensor<f32> = Tensor::rand([j, k]);

    let start = ProcessTime::now();
    _ = einsum(&[&tensor_a, &tensor_b], (["ij", "jk"], "ij"));
    start.elapsed().as_nanos()
}

fn einsum_2operands_3() -> u128 {
    let i = 100;
    let j = 50;
    let k = 100;

    let tensor_a: Tensor<f32> = Tensor::rand([i, j]);
    let tensor_b: Tensor<f32> = Tensor::rand([j, k]);

    let start = ProcessTime::now();
    _ = einsum(&[&tensor_a, &tensor_b], (["ij", "jk"], "ijk"));
    start.elapsed().as_nanos()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let id = args[1].parse::<usize>().unwrap();

    let time =
        if id == 1 { einsum1() }
        else if id == 2 { einsum2() }
        else if id == 3 { einsum3() }
        else if id == 4 { einsum4() }
        else if id == 5 { einsum5() }
        else if id == 6 { einsum6() }
        else if id == 7 { einsum7() }
        else if id == 8 { einsum8() }
        else if id == 9 { einsum9() }
        else if id == 10 { einsum10() }

        else if id == 100 { einsum_2operands_0() }
        else if id == 101 { einsum_2operands_1() }
        else if id == 102 { einsum_2operands_2() }
        else if id == 103 { einsum_2operands_3() }

        else { panic!("invalid ID") };

    println!("{}", time);
}
