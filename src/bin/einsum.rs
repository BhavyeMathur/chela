use chela::*;
use std::env;

use cpu_time::ProcessTime;


const I: usize = 100;
const J: usize = 500;
const K: usize = 1000;

const U: usize = 1000;
const V: usize = 500;


fn einsum1() -> u128 {
    let i = 10000;

    let tensor_a: Tensor<f32> = Tensor::rand([i]);
    let tensor_b: Tensor<f32> = Tensor::rand([i]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["i", "i"], ""));
    start.elapsed().as_nanos()
}

fn einsum2() -> u128 {
    let i = 1000;
    let j = 500;

    let tensor_a: Tensor<f32> = Tensor::rand([i, j]);
    let tensor_b: Tensor<f32> = Tensor::rand([j]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "j"], "i"));
    start.elapsed().as_nanos()
}

fn einsum3() -> u128 {
    let i = 100;
    let j = 1000;
    let k = 500;

    let tensor_a: Tensor<f32> = Tensor::rand([i, j]);
    let tensor_b: Tensor<f32> = Tensor::rand([j, k]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], "ik"));
    start.elapsed().as_nanos()
}

fn einsum4() -> u128 {
    let i = 100;
    let j = 1000;
    let k = 500;

    let tensor_a: Tensor<f32> = Tensor::rand([i, k]);
    let tensor_b: Tensor<f32> = Tensor::rand([j, k]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ik", "jk"], "ij"));
    start.elapsed().as_nanos()
}

fn einsum5() -> u128 {
    let i = 100;
    let j = 500;
    let k = 100;
    let l = 3;

    let tensor_a: Tensor<f32> = Tensor::rand([i, j, l]);
    let tensor_b: Tensor<f32> = Tensor::rand([j, k, l]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ijl", "jkl"], "ikl"));
    start.elapsed().as_nanos()
}

fn einsum6() -> u128 {
    let i = 1000;

    let tensor_a: Tensor<f32> = Tensor::rand([i, i]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a], (["ii"], ""));
    start.elapsed().as_nanos()
}

fn einsum7() -> u128 {
    let i = 128;
    let j = 64;
    let k = 32;

    let tensor_a: Tensor<f32> = Tensor::rand([i, j]);
    let tensor_b: Tensor<f32> = Tensor::rand([k, j]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "kj"], "ikj"));
    start.elapsed().as_nanos()
}

fn einsum8() -> u128 {
    let a = 128;
    let b = 64;
    let c = 32;

    let tensor_a: Tensor<f32> = Tensor::rand([a, b, c]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a], (["abc"], ""));
    start.elapsed().as_nanos()
}

fn einsum9() -> u128 {
    let i = 1000;

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

fn einsum11() -> u128 {
    let i = 100;
    let j = 100;
    let k = 100;

    let tensor_a: Tensor<f32> = Tensor::rand([i, j, k]);
    let tensor_b: Tensor<f32> = Tensor::rand([i, j, k]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ijk", "ijk"], "ijk"));
    start.elapsed().as_nanos()
}

fn einsum_2operands_0() -> u128 {
    let tensor_a: Tensor<f32> = Tensor::rand([I, J]);
    let tensor_b: Tensor<f32> = Tensor::rand([J, K]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], ""));
    start.elapsed().as_nanos()
}

fn einsum_2operands_1() -> u128 {
    let tensor_a: Tensor<f32> = Tensor::rand([I, J]);
    let tensor_b: Tensor<f32> = Tensor::rand([J, K]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], "i"));
    start.elapsed().as_nanos()
}

fn einsum_2operands_2() -> u128 {
    let tensor_a: Tensor<f32> = Tensor::rand([I, J]);
    let tensor_b: Tensor<f32> = Tensor::rand([J, K]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], "ij"));
    start.elapsed().as_nanos()
}

fn einsum_2operands_3() -> u128 {
    let tensor_a: Tensor<f32> = Tensor::rand([I, J]);
    let tensor_b: Tensor<f32> = Tensor::rand([J, K]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], "ijk"));
    start.elapsed().as_nanos()
}

fn einsum_2operands_4() -> u128 {
    let tensor_a: Tensor<f32> = Tensor::rand([I, J]);
    let tensor_b: Tensor<f32> = Tensor::rand([J, K]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], "ik"));
    start.elapsed().as_nanos()
}

fn einsum_2operands_5() -> u128 {
    let tensor_a: Tensor<f32> = Tensor::rand([I, K]);
    let tensor_b: Tensor<f32> = Tensor::rand([J, K]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ik", "jk"], "ij"));
    start.elapsed().as_nanos()
}

fn einsum_2operands_6() -> u128 {
    let tensor_a: Tensor<f32> = Tensor::rand([I, J]);
    let tensor_b: Tensor<f32> = Tensor::rand([K, I]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "ki"], "j"));
    start.elapsed().as_nanos()
}

fn einsum_2operands_7() -> u128 {
    let tensor_a: Tensor<f32> = Tensor::rand([I, J]);
    let tensor_b: Tensor<f32> = Tensor::rand([K, I]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "ki"], "i"));
    start.elapsed().as_nanos()
}

fn einsum_2operands_8() -> u128 {
    let tensor_a: Tensor<f32> = Tensor::rand([I, J]);
    let tensor_b: Tensor<f32> = Tensor::rand([J]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "j"], "i"));
    start.elapsed().as_nanos()
}

fn einsum_on_slices0() -> u128 {
    let tensor_a: Tensor<f32> = Tensor::rand([I, J]);
    let tensor_b: Tensor<f32> = Tensor::rand([J, K, 2]);

    let tensor_b = tensor_b.slice_along(Axis(2), 0);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], ""));
    start.elapsed().as_nanos()
}

fn einsum_on_slices1() -> u128 {
    let tensor_a: Tensor<f32> = Tensor::rand([10000, 2]);
    let tensor_a = tensor_a.slice_along(Axis(1), 0);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a], (["i"], ""));
    start.elapsed().as_nanos()
}

fn einsum_3operands_0() -> u128 {
    let tensor_a: Tensor<f32> = Tensor::rand([I]);
    let tensor_b: Tensor<f32> = Tensor::rand([J]);
    let tensor_c: Tensor<f32> = Tensor::rand([K]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b, &tensor_c], (["i", "j", "k"], ""));
    start.elapsed().as_nanos()
}

fn einsum_3operands_1() -> u128 {
    let tensor_a: Tensor<f32> = Tensor::rand([I, J]);
    let tensor_b: Tensor<f32> = Tensor::rand([J]);
    let tensor_c: Tensor<f32> = Tensor::rand([K]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b, &tensor_c], (["ij", "j", "k"], ""));
    start.elapsed().as_nanos()
}

fn einsum_4operands_0() -> u128 {
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
    _ = einsum([&tensor_a, &tensor_b, &tensor_c, &tensor_d], (["abc", "bd", "bc", "de"], "ae"));
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
        else if id == 11 { einsum11() }

        else if id == 100 { einsum_2operands_0() }
        else if id == 101 { einsum_2operands_1() }
        else if id == 102 { einsum_2operands_2() }
        else if id == 103 { einsum_2operands_3() }
        else if id == 104 { einsum_2operands_4() }
        else if id == 105 { einsum_2operands_5() }
        else if id == 106 { einsum_2operands_6() }
        else if id == 107 { einsum_2operands_7() }
        else if id == 108 { einsum_2operands_8() }

        else if id == 200 { einsum_on_slices0() }
        else if id == 201 { einsum_on_slices1() }
        else if id == 202 { einsum_3operands_0() }
        else if id == 203 { einsum_3operands_1() }
        else if id == 204 { einsum_4operands_0() }

        else { panic!("invalid ID") };

    println!("{}", time);
}
