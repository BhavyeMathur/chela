use chela::*;
use std::env;

use cpu_time::ProcessTime;


type T = f32;


fn backward0() -> u128 {
    let i = 10000;

    let tensor_a = NdArray::<T>::rand([i]);
    let tensor_b = NdArray::<T>::rand([i]);

    let start = ProcessTime::now();
    _ = einsum([&tensor_a, &tensor_b], (["i", "i"], ""));
    start.elapsed().as_nanos()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let id = args[1].parse::<usize>().unwrap();

    let time =
        if id == 0 { backward0() }

        else { panic!("invalid ID") };

    println!("{}", time);
}
