pub(super) mod util;

pub mod gradient_function;

pub mod none_backwards;
pub mod identity_backwards;
pub mod accumulate_grad;

pub mod add_backwards;
pub mod sub_backwards;
pub mod mul_backwards;
pub mod div_backwards;
pub mod neg_backwards;

pub mod dot_backwards;
pub mod matrix_vec_backwards;
pub mod matrix_product_backwards;
pub mod bmm_backwards;

pub mod reshape_backwards;
pub mod transpose_backwards;