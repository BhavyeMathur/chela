use redstone_ml::*;

fn main() {
    let _grad = {
        let mut a = Tensor::new([1.0f32, 2.0, 3.0]);
        a.set_requires_grad(true);
        a.backward();

        a.gradient().unwrap()
    };
}
