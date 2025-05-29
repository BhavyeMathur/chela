use chela::*;

fn main() {
    let _grad = {
        let mut a = NdArray::from([1.0f32, 2.0, 3.0]);
        a.set_requires_grad(true);
        a.backward();

        a.gradient().unwrap()
    };
}
