use chela::*;
use std::env;

use chela::profiler::*;

type T = i32;
const K: usize = 4096;

fn binary_ops10() {
    let tensor1 = NdArray::<f32>::rand([K]).astype::<T>();
    let tensor2 = NdArray::<f32>::rand([K]).astype::<T>();

    let func = || { _ = &tensor1 + &tensor2; };
    profile_func(func)
}


fn binary_ops0() {
    let tensor1 = NdArray::<f32>::rand([K]).astype::<T>();
    let tensor2 = NdArray::<f32>::rand([K]).astype::<T>();

    let func = || { _ = &tensor1 + &tensor2; };
    profile_func(func)
}

fn binary_ops1() {
    let tensor1 = NdArray::<f32>::rand([K]).astype::<T>();
    let tensor2 = NdArray::<f32>::rand([1]).astype::<T>();

    let func = || { _ = &tensor1 + &tensor2; };
    profile_func(func)
}

fn binary_ops2() {
    let tensor1: NdArray<'static, T> = NdArray::<f32>::rand([K, 3]).astype::<T>();
    let tensor1 = tensor1.slice_along(Axis(-1), 0);

    let tensor2 = NdArray::<f32>::rand([K]).astype::<T>();

    let func = || { _ = &tensor1 + &tensor2; };
    profile_func(func)
}


fn binary_ops3() {
    let tensor1 = NdArray::<f32>::rand([K, 3]).astype::<T>();
    let tensor1 = tensor1.slice_along(Axis(-1), 0);

    let tensor2 = NdArray::<f32>::rand([1]).astype::<T>();

    let func = || { _ = &tensor1 + &tensor2; };
    profile_func(func)
}

fn binary_ops4() {
    let tensor1 = NdArray::<f32>::rand([K, 3]).astype::<T>();
    let tensor1 = tensor1.slice_along(Axis(-1), 0..2);

    let tensor2 = NdArray::<f32>::rand([K, 2]).astype::<T>();

    let func = || { _ = &tensor1 + &tensor2; };
    profile_func(func)
}

fn binary_ops5() {
    let tensor1 = NdArray::<f32>::rand([K, 3]).astype::<T>();
    let tensor1 = tensor1.slice_along(Axis(-1), 0..2);

    let tensor2 = NdArray::<f32>::rand([1]).astype::<T>();

    let func = || { _ = &tensor1 + &tensor2; };
    profile_func(func)
}

fn binary_ops6() {
    let tensor1 = NdArray::<f32>::rand([K, 3]).astype::<T>();
    let tensor1 = tensor1.slice_along(Axis(-1), 0..2);

    let tensor2 = NdArray::<f32>::rand([K, 2, 2]).astype::<T>();
    let tensor2 = tensor2.slice_along(Axis(-1), 0);

    let func = || { _ = &tensor1 + &tensor2; };
    profile_func(func)
}

fn binary_ops7() {
    let tensor1 = NdArray::<f32>::rand([K, 3]).astype::<T>();
    let tensor1 = tensor1.slice_along(Axis(-1), 0..2);

    let tensor2 = NdArray::<f32>::rand([K, 3]).astype::<T>();
    let tensor2 = tensor2.slice_along(Axis(-1), 0..2);

    let func = || { _ = &tensor1 + &tensor2; };
    profile_func(func)
}

fn binary_ops8() {
    let tensor1 = NdArray::<f32>::rand([K, 3]).astype::<T>();
    let tensor1 = tensor1.slice_along(Axis(-1), 0);

    let tensor2 = NdArray::<f32>::rand([K, 3]).astype::<T>();
    let tensor2 = tensor2.slice_along(Axis(-1), 0);

    let func = || { _ = &tensor1 + &tensor2; };
    profile_func(func)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let test_id = args[1].parse::<usize>().unwrap();

    match test_id {
        0 => { binary_ops0() },
        1 => { binary_ops1() },
        2 => { binary_ops2() },
        3 => { binary_ops3() },
        4 => { binary_ops4() },
        5 => { binary_ops5() },
        6 => { binary_ops6() },
        7 => { binary_ops7() },
        8 => { binary_ops8() },
        _ => { panic!("invalid ID") },
    }
}
