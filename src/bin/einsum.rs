use chela::profiler::profile_func;
use chela::*;
use std::env;

const I: usize = 100;
const J: usize = 500;
const K: usize = 1000;

const U: usize = 1000;
const V: usize = 500;

type T = f32;


fn einsum1() {
    let i = 10000;

    let tensor_a = NdArray::<f32>::rand([i]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([i]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b], (["i", "i"], "")); };
    profile_func(func)
}

fn einsum1001() {
    let i = 10000;

    let tensor_a = NdArray::<f32>::rand([i]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([i]).astype::<T>();
    
    let func = || { _ = tensor_a.dot(&tensor_b); };
    profile_func(func)
}

fn einsum2() {
    let i = 1000;
    let j = 500;

    let tensor_a = NdArray::<f32>::rand([i, j]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([j]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b], (["ij", "j"], "i")); };
    profile_func(func)
}

fn einsum1002() {
    let i = 1000;
    let j = 500;

    let tensor_a = NdArray::<f32>::rand([i, j]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([j]).astype::<T>();
    
    let func = || { _ = tensor_a.matmul(&tensor_b); };
    profile_func(func)
}

fn einsum3() {
    let i = 100;
    let j = 1000;
    let k = 500;

    let tensor_a = NdArray::<f32>::rand([i, j]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([j, k]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], "ik")); };
    profile_func(func)
}

fn einsum1003() {
    let i = 100;
    let j = 1000;
    let k = 500;

    let tensor_a = NdArray::<f32>::rand([i, j]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([j, k]).astype::<T>();
    
    let func = || { _ = tensor_a.matmul(&tensor_b); };
    profile_func(func)
}

fn einsum4() {
    let i = 100;
    let j = 1000;
    let k = 500;

    let tensor_a = NdArray::<f32>::rand([i, k]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([j, k]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b], (["ik", "jk"], "ij")); };
    profile_func(func)
}

fn einsum5() {
    let i = 100;
    let j = 50;
    let k = 100;
    let b = 64;

    let tensor_a = NdArray::<f32>::rand([b, i, j]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([b, j, k]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b], (["bij", "bjk"], "bik")); };
    profile_func(func)
}

fn einsum1005() {
    let i = 100;
    let j = 50;
    let k = 100;
    let b = 64;

    let tensor_a = NdArray::<f32>::rand([b, i, j]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([b, j, k]).astype::<T>();
    
    let func = || { _ = tensor_a.bmm(&tensor_b); };
    profile_func(func)
}

fn einsum6() {
    let i = 1000;

    let tensor_a = NdArray::<f32>::rand([i, i]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a], (["ii"], "")); };
    profile_func(func)
}

fn einsum1006() {
    let i = 1000;

    let tensor_a = NdArray::<f32>::rand([i, i]).astype::<T>();
    
    let func = || { _ = tensor_a.trace(); };
    profile_func(func)
}

fn einsum7() {
    let i = 128;
    let j = 64;
    let k = 32;

    let tensor_a = NdArray::<f32>::rand([i, j]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([k, j]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b], (["ij", "kj"], "ikj")); };
    profile_func(func)
}

fn einsum8() {
    let a = 128;
    let b = 64;
    let c = 32;

    let tensor_a = NdArray::<f32>::rand([a, b, c]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a], (["abc"], "")); };
    profile_func(func)
}

fn einsum9() {
    let i = 1000;

    let tensor_a = NdArray::<f32>::rand([i, i]).astype::<T>();
    
    let func = || { _ = einsum_view(&tensor_a, ("ii", "i")).unwrap(); };
    profile_func(func)
}

fn einsum1009() {
    let i = 1000;

    let tensor_a = NdArray::<f32>::rand([i, i]).astype::<T>();
    
    let func = || { _ = tensor_a.diagonal(); };
    profile_func(func)
}

fn einsum10() {
    let a = 10;
    let b = 20;
    let c = 30;
    let d = 40;

    let tensor_a = NdArray::<f32>::rand([a, b, c, d]).astype::<T>();
    
    let func = || { _ = einsum_view(&tensor_a, ("abcd", "dcba")).unwrap(); };
    profile_func(func)
}

fn einsum11() {
    let i = 100;
    let j = 100;
    let k = 100;

    let tensor_a = NdArray::<f32>::rand([i, j, k]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([i, j, k]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b], (["ijk", "ijk"], "ijk")); };
    profile_func(func)
}

fn einsum12() {
    let i = 100;
    let j = 1000;

    let tensor_a = NdArray::<f32>::rand([i]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([j]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b], (["i", "j"], "ij")); };
    profile_func(func)
}

fn einsum13() {
    let b = 512;
    let i = 1000;

    let tensor_a = NdArray::<f32>::rand([b, i]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([b, i]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b], (["bi", "bi"], "b")); };
    profile_func(func)
}

fn einsum14() {
    let i = 100;
    let j = 90;
    let k = 150;
    let l = 50;

    let tensor_a = NdArray::<f32>::rand([i, j]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([i, k]).astype::<T>();
    let tensor_c = NdArray::<f32>::rand([i, l]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b, &tensor_c], (["ij", "ik", "il"], "jkl")); };
    profile_func(func)
}

fn einsum_2operands_0() {
    let tensor_a = NdArray::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([J, K]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], "")); };
    profile_func(func)
}

fn einsum_2operands_1() {
    let tensor_a = NdArray::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([J, K]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], "i")); };
    profile_func(func)
}

fn einsum_2operands_2() {
    let tensor_a = NdArray::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([J, K]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], "ij")); };
    profile_func(func)
}

fn einsum_2operands_3() {
    let tensor_a = NdArray::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([J, K]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], "ijk")); };
    profile_func(func)
}

fn einsum_2operands_4() {
    let tensor_a = NdArray::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([J, K]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], "ik")); };
    profile_func(func)
}

fn einsum_2operands_5() {
    let tensor_a = NdArray::<f32>::rand([I, K]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([J, K]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b], (["ik", "jk"], "ij")); };
    profile_func(func)
}

fn einsum_2operands_6() {
    let tensor_a = NdArray::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([K, I]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b], (["ij", "ki"], "j")); };
    profile_func(func)
}

fn einsum_2operands_7() {
    let tensor_a = NdArray::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([K, I]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b], (["ij", "ki"], "i")); };
    profile_func(func)
}

fn einsum_2operands_8() {
    let tensor_a = NdArray::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([J]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b], (["ij", "j"], "i")); };
    profile_func(func)
}

fn einsum_on_slices0() {
    let tensor_a = NdArray::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([J, K, 2]).astype::<T>();

    let tensor_b = tensor_b.slice_along(Axis(2), 0);

    let func = || { _ = einsum([&tensor_a, &tensor_b], (["ij", "jk"], "")); };
    profile_func(func)
}

fn einsum_on_slices1() {
    let tensor_a = NdArray::<f32>::rand([10000, 2]).astype::<T>();
    let tensor_a = tensor_a.slice_along(Axis(1), 0);

    let func = || { _ = einsum([&tensor_a], (["i"], "")); };
    profile_func(func)
}

fn einsum_3operands_0() {
    let tensor_a = NdArray::<f32>::rand([I]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([J]).astype::<T>();
    let tensor_c = NdArray::<f32>::rand([K]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b, &tensor_c], (["i", "j", "k"], "")); };
    profile_func(func)
}

fn einsum_3operands_1() {
    let tensor_a = NdArray::<f32>::rand([I, J]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([J]).astype::<T>();
    let tensor_c = NdArray::<f32>::rand([K]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b, &tensor_c], (["ij", "j", "k"], "")); };
    profile_func(func)
}

fn einsum_4operands_0() {
    let a = 100;
    let b = 5;
    let c = 20;
    let d = 50;
    let e = 100;

    let tensor_a = NdArray::<f32>::rand([a, b, c]).astype::<T>();
    let tensor_b = NdArray::<f32>::rand([b, d]).astype::<T>();
    let tensor_c = NdArray::<f32>::rand([b, c]).astype::<T>();
    let tensor_d = NdArray::<f32>::rand([d, e]).astype::<T>();
    
    let func = || { _ = einsum([&tensor_a, &tensor_b, &tensor_c, &tensor_d], (["abc", "bd", "bc", "de"], "ae")); };
    profile_func(func)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let id = args[1].parse::<usize>().unwrap();

    match id {
        1 => { einsum1() },
        2 => { einsum2() },
        3 => { einsum3() },
        4 => { einsum4() },
        5 => { einsum5() },
        6 => { einsum6() },
        7 => { einsum7() },
        8 => { einsum8() },
        9 => { einsum9() },
        10 => { einsum10() },
        11 => { einsum11() },
        12 => { einsum12() },
        13 => { einsum13() },
        14 => { einsum14() },

        100 => { einsum_2operands_0() },
        101 => { einsum_2operands_1() },
        102 => { einsum_2operands_2() },
        103 => { einsum_2operands_3() },
        104 => { einsum_2operands_4() },
        105 => { einsum_2operands_5() },
        106 => { einsum_2operands_6() },
        107 => { einsum_2operands_7() },
        108 => { einsum_2operands_8() },

        200 => { einsum_on_slices0() },
        201 => { einsum_on_slices1() },
        202 => { einsum_3operands_0() },
        203 => { einsum_3operands_1() },
        204 => { einsum_4operands_0() },

        1001 => { einsum1001() },
        1002 => { einsum1002() },
        1003 => { einsum1003() },
        1005 => { einsum1005() },
        1006 => { einsum1006() },
        1009 => { einsum1009() },

        _ => { panic!("Invalid id"); }
    }
}
