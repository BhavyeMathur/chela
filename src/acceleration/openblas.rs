#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

type CBLAS_ORDER = i32;
type CBLAS_TRANSPOSE = i32;
type blasint = i32;
type float = f32;


#[cfg(openblas)]
extern "C" {
    pub(crate) fn cblas_sgemm_batch(Order: CBLAS_ORDER,
                             TransA_array: *const CBLAS_TRANSPOSE,
                             TransB_array: *const CBLAS_TRANSPOSE,
                             M_array: *const blasint,
                             N_array: *const blasint,
                             K_array: *const blasint,
                             alpha_array: *const float,
                             A_array: *const *const float,
                             lda_array: *const blasint,
                             B_array: *const *const float,
                             ldb_array: *const blasint,
                             beta_array: *const float,
                             C_array: *mut *mut float,
                             ldc_array: *const blasint,
                             group_count: blasint,
                             group_size: *const blasint,
    );
}
