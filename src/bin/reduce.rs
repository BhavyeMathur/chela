use redstone::*;
use std::env;

use redstone::profiler::profile_func;

const N: usize = 1000000;

type T = f32;


fn reduce0() {
    let tensor = NdArray::<f32>::rand([N]).astype::<T>();

    let func = || { _ = tensor.sum(); };
    profile_func(func)
}

fn reduce1() {
    let tensor = NdArray::<f32>::rand([N]).astype::<T>();

    let func = || { _ = tensor.product(); };
    profile_func(func)
}

fn reduce2() {
    let tensor = NdArray::<f32>::rand([N]).astype::<T>();

    let func = || { _ = tensor.min(); };
    profile_func(func)
}

fn reduce3() {
    let tensor = NdArray::<f32>::rand([N]).astype::<T>();

    let func = || { _ = tensor.max(); };
    profile_func(func)
}

fn reduce10() {
    let tensor = NdArray::<f32>::rand([N, 2]).astype::<T>();
    let tensor = tensor.slice_along(Axis(1), 0);

    let func = || { _ = tensor.sum(); };
    profile_func(func)
}

fn reduce11() {
    let tensor = NdArray::<f32>::rand([N, 2]).astype::<T>();
    let tensor = tensor.slice_along(Axis(1), 0);

    let func = || { _ = tensor.product(); };
    profile_func(func)
}

fn reduce12() {
    let tensor = NdArray::<f32>::rand([N, 2]).astype::<T>();
    let tensor = tensor.slice_along(Axis(1), 0);

    let func = || { _ = tensor.min(); };
    profile_func(func)
}

fn reduce13() {
    let tensor = NdArray::<f32>::rand([N, 2]).astype::<T>();
    let tensor = tensor.slice_along(Axis(1), 0);

    let func = || { _ = tensor.max(); };
    profile_func(func)
}

fn reduce20() {
    let tensor = NdArray::<f32>::rand([N, 3]).astype::<T>();
    let tensor = tensor.slice_along(Axis(1), 0..2);

    let func = || { _ = tensor.sum(); };
    profile_func(func)
}

fn reduce21() {
    let tensor = NdArray::<f32>::rand([N, 3]).astype::<T>();
    let tensor = tensor.slice_along(Axis(1), 0..2);

    let func = || { _ = tensor.product(); };
    profile_func(func)
}

fn reduce22() {
    let tensor = NdArray::<f32>::rand([N, 3]).astype::<T>();
    let tensor = tensor.slice_along(Axis(1), 0..2);

    let func = || { _ = tensor.min(); };
    profile_func(func)
}

fn reduce23() {
    let tensor = NdArray::<f32>::rand([N, 3]).astype::<T>();
    let tensor = tensor.slice_along(Axis(1), 0..2);

    let func = || { _ = tensor.max(); };
    profile_func(func)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let id = args[1].parse::<usize>().unwrap();

    match id {
        0 => { reduce0() },
        1 => { reduce1() },
        2 => { reduce2() },
        3 => { reduce3() },

        10 => { reduce10() },
        11 => { reduce11() },
        12 => { reduce12() },
        13 => { reduce13() },

        20 => { reduce20() },
        21 => { reduce21() },
        22 => { reduce22() },
        23 => { reduce23() },

        _ => { panic!("invalid ID") }
    }
}
