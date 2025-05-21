use chela::*;
use std::env;

use cpu_time::ProcessTime;


const I: usize = 100;
const J: usize = 500;
const K: usize = 1000;

const U: usize = 1000;
const V: usize = 500;

type T = f32;


fn einsum1() -> u128 {
    let i = 10000;

    let tensor_a = Tensor::<f32>::rand([i]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([i]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["i", "i"], ""));
    start.elapsed().as_nanos()
}

fn einsum2() -> u128 {
    let i = 1000;
    let j = 500;

    let tensor_a = Tensor::<f32>::rand([i, j]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([j]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "j"], "i"));
    start.elapsed().as_nanos()
}

fn einsum3() -> u128 {
    let i = 100;
    let j = 1000;
    let k = 500;

    let tensor_a = Tensor::<f32>::rand([i, j]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([j, k]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], "ik"));
    start.elapsed().as_nanos()
}

fn einsum4() -> u128 {
    let i = 100;
    let j = 1000;
    let k = 500;

    let tensor_a = Tensor::<f32>::rand([i, k]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([j, k]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ik", "jk"], "ij"));
    start.elapsed().as_nanos()
}

fn einsum5() -> u128 {
    let i = 100;
    let j = 50;
    let k = 100;
    let b = 64;

    let tensor_a = Tensor::<f32>::rand([b, i, j]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([b, j, k]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["bij", "bjk"], "bik"));
    start.elapsed().as_nanos()
}

fn einsum6() -> u128 {
    let i = 1000;

    let tensor_a = Tensor::<f32>::rand([i, i]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a], (["ii"], ""));
    start.elapsed().as_nanos()
}

fn einsum1006() -> u128 {
    let i = 1000;

    let tensor_a = Tensor::<f32>::rand([i, i]).astype::<T>();

    let start = ProcessTime::now();
    _ = tensor_a.trace();
    start.elapsed().as_nanos()
}

fn einsum7() -> u128 {
    let i = 128;
    let j = 64;
    let k = 32;

    let tensor_a = Tensor::<f32>::rand([i, j]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([k, j]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "kj"], "ikj"));
    start.elapsed().as_nanos()
}

fn einsum8() -> u128 {
    let a = 128;
    let b = 64;
    let c = 32;

    let tensor_a = Tensor::<f32>::rand([a, b, c]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a], (["abc"], ""));
    start.elapsed().as_nanos()
}

fn einsum9() -> u128 {
    let i = 1000;

    let tensor_a = Tensor::<f32>::rand([i, i]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum_view(&tensor_a, ("ii", "i")).unwrap();
    start.elapsed().as_nanos()
}

fn einsum1009() -> u128 {
    let i = 1000;

    let tensor_a = Tensor::<f32>::rand([i, i]).astype::<T>();

    let start = ProcessTime::now();
    _ = tensor_a.diagonal();
    start.elapsed().as_nanos()
}

fn einsum10() -> u128 {
    let a = 10;
    let b = 20;
    let c = 30;
    let d = 40;

    let tensor_a = Tensor::<f32>::rand([a, b, c, d]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum_view(&tensor_a, ("abcd", "dcba")).unwrap();
    start.elapsed().as_nanos()
}

fn einsum11() -> u128 {
    let i = 100;
    let j = 100;
    let k = 100;

    let tensor_a = Tensor::<f32>::rand([i, j, k]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([i, j, k]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ijk", "ijk"], "ijk"));
    start.elapsed().as_nanos()
}

fn einsum12() -> u128 {
    let i = 100;
    let j = 1000;

    let tensor_a = Tensor::<f32>::rand([i]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([j]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["i", "j"], "ij"));
    start.elapsed().as_nanos()
}

fn einsum13() -> u128 {
    let b = 512;
    let i = 1000;

    let tensor_a = Tensor::<f32>::rand([b, i]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([b, i]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["bi", "bi"], "b"));
    start.elapsed().as_nanos()
}

fn einsum14() -> u128 {
    let i = 100;
    let j = 90;
    let k = 150;
    let l = 50;

    let tensor_a = Tensor::<f32>::rand([i, j]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([i, k]).astype::<T>();
    let tensor_c = Tensor::<f32>::rand([i, l]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b, &tensor_c], (["ij", "ik", "il"], "jkl"));
    start.elapsed().as_nanos()
}

fn einsum_2operands_0() -> u128 {
    let tensor_a = Tensor::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([J, K]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], ""));
    start.elapsed().as_nanos()
}

fn einsum_2operands_1() -> u128 {
    let tensor_a = Tensor::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([J, K]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], "i"));
    start.elapsed().as_nanos()
}

fn einsum_2operands_2() -> u128 {
    let tensor_a = Tensor::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([J, K]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], "ij"));
    start.elapsed().as_nanos()
}

fn einsum_2operands_3() -> u128 {
    let tensor_a = Tensor::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([J, K]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], "ijk"));
    start.elapsed().as_nanos()
}

fn einsum_2operands_4() -> u128 {
    let tensor_a = Tensor::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([J, K]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], "ik"));
    start.elapsed().as_nanos()
}

fn einsum_2operands_5() -> u128 {
    let tensor_a = Tensor::<f32>::rand([I, K]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([J, K]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ik", "jk"], "ij"));
    start.elapsed().as_nanos()
}

fn einsum_2operands_6() -> u128 {
    let tensor_a = Tensor::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([K, I]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "ki"], "j"));
    start.elapsed().as_nanos()
}

fn einsum_2operands_7() -> u128 {
    let tensor_a = Tensor::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([K, I]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "ki"], "i"));
    start.elapsed().as_nanos()
}

fn einsum_2operands_8() -> u128 {
    let tensor_a = Tensor::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([J]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "j"], "i"));
    start.elapsed().as_nanos()
}

fn einsum_on_slices0() -> u128 {
    let tensor_a = Tensor::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([J, K, 2]).astype::<T>();

    let tensor_b = tensor_b.slice_along(Axis(2), 0);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], ""));
    start.elapsed().as_nanos()
}

fn einsum_on_slices1() -> u128 {
    let tensor_a = Tensor::<f32>::rand([10000, 2]).astype::<T>();
    let tensor_a = tensor_a.slice_along(Axis(1), 0);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a], (["i"], ""));
    start.elapsed().as_nanos()
}

fn einsum_3operands_0() -> u128 {
    let tensor_a = Tensor::<f32>::rand([I]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([J]).astype::<T>();
    let tensor_c = Tensor::<f32>::rand([K]).astype::<T>();

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b, &tensor_c], (["i", "j", "k"], ""));
    start.elapsed().as_nanos()
}

fn einsum_3operands_1() -> u128 {
    let tensor_a = Tensor::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([J]).astype::<T>();
    let tensor_c = Tensor::<f32>::rand([K]).astype::<T>();

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

    let tensor_a = Tensor::<f32>::rand([a, b, c]).astype::<T>();
    let tensor_b = Tensor::<f32>::rand([b, d]).astype::<T>();
    let tensor_c = Tensor::<f32>::rand([b, c]).astype::<T>();
    let tensor_d = Tensor::<f32>::rand([d, e]).astype::<T>();

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
        else if id == 12 { einsum12() }
        else if id == 13 { einsum13() }
        else if id == 14 { einsum14() }

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

        else if id == 1006 { einsum1006() }
        else if id == 1009 { einsum1009() }

        else { panic!("invalid ID") };

    println!("{}", time);
}
