pub(super) mod util;
pub(crate) mod into;

pub mod gradient_function;

pub mod none_backwards;
pub mod accumulate_grad;

pub mod add_backwards;
pub mod sub_backwards;
pub mod mul_backwards;
pub mod div_backwards;
pub mod neg_backwards;

pub mod backwards;