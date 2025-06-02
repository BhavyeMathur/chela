#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

#[cfg(apple_vdsp)]
use std::ffi::{c_double, c_float, c_int};

type vDSP_Length = usize;
type vDSP_Stride = isize;

#[cfg(apple_vdsp)]
#[link(name = "Accelerate")]
extern "C" {
    
    // element-wise addition
    pub(crate) fn vDSP_vadd(__A: *const c_float, __IA: isize,
                            __B: *const c_float, __IB: isize,
                            __C: *mut c_float, __IC: vDSP_Stride, __N: vDSP_Length);

    pub(crate) fn vDSP_vaddD(__A: *const c_double, __IA: isize,
                             __B: *const c_double, __IB: isize,
                             __C: *mut c_double, __IC: vDSP_Stride, __N: vDSP_Length);

    pub(crate) fn vDSP_vaddi(__A: *const c_int, __IA: isize,
                             __B: *const c_int, __IB: isize,
                             __C: *mut c_int, __IC: vDSP_Stride, __N: vDSP_Length);

    
    // vector-scalar addition
    pub(crate) fn vDSP_vsadd(__A: *const c_float, __IA: vDSP_Stride,
                             __B: *const c_float,
                             __C: *mut c_float, __IC: vDSP_Stride,
                             __N: vDSP_Length,
    );

    pub(crate) fn vDSP_vsaddD(__A: *const c_double, __IA: vDSP_Stride,
                              __B: *const c_double,
                              __C: *mut c_double, __IC: vDSP_Stride,
                              __N: vDSP_Length,
    );

    // element-wise subtraction
    pub(crate) fn vDSP_vsub(__A: *const c_float, __IA: isize,
                            __B: *const c_float, __IB: isize,
                            __C: *mut c_float, __IC: vDSP_Stride, __N: vDSP_Length);

    pub(crate) fn vDSP_vsubD(__A: *const c_double, __IA: isize,
                             __B: *const c_double, __IB: isize,
                             __C: *mut c_double, __IC: vDSP_Stride, __N: vDSP_Length);

    
    // element-wise multiplication
    pub(crate) fn vDSP_vmul(__A: *const c_float, __IA: isize,
                            __B: *const c_float, __IB: isize,
                            __C: *mut c_float, __IC: vDSP_Stride, __N: vDSP_Length);

    pub(crate) fn vDSP_vmulD(__A: *const c_double, __IA: isize,
                             __B: *const c_double, __IB: isize,
                             __C: *mut c_double, __IC: vDSP_Stride, __N: vDSP_Length);

    
    // vector-scalar multiplication
    pub(crate) fn vDSP_vsmul(__A: *const c_float, __IA: vDSP_Stride,
                             __B: *const c_float,
                             __C: *mut c_float, __IC: vDSP_Stride,
                             __N: vDSP_Length,
    );

    pub(crate) fn vDSP_vsmulD(__A: *const c_double, __IA: vDSP_Stride,
                              __B: *const c_double,
                              __C: *mut c_double, __IC: vDSP_Stride,
                              __N: vDSP_Length,
    );

    pub(crate) fn vDSP_vsaddi(__A: *const c_int, __IA: vDSP_Stride,
                              __B: *const c_int,
                              __C: *mut c_int, __IC: vDSP_Stride,
                              __N: vDSP_Length,
    );

    
    // vector fill
    pub(crate) fn vDSP_vfill(__A: *const c_float,
                             __C: *mut c_float, __IC: vDSP_Stride,
                             __N: vDSP_Length);

    pub(crate) fn vDSP_vfillD(__A: *const c_double,
                              __C: *mut c_double, __IC: vDSP_Stride,
                              __N: vDSP_Length);

    pub(crate) fn vDSP_vfilli(__A: *const c_int,
                              __C: *mut c_int, __IC: vDSP_Stride,
                              __N: vDSP_Length);

    
    // vector sum
    pub(crate) fn vDSP_sve(__A: *const c_float, __I: isize, __C: *mut c_float, __N: vDSP_Length);

    pub(crate) fn vDSP_sveD(__A: *const c_double, __I: isize, __C: *mut c_double, __N: vDSP_Length);

    
    // vector minimum & maximum
    pub(crate) fn vDSP_maxv(__A: *const c_float, __IA: isize,
                            __C: *mut c_float, __N: vDSP_Length);

    pub(crate) fn vDSP_maxvD(__A: *const c_double, __IA: isize,
                             __C: *mut c_double, __N: vDSP_Length);

    pub(crate) fn vDSP_minv(__A: *const c_float, __IA: isize,
                            __C: *mut c_float, __N: vDSP_Length);

    pub(crate) fn vDSP_minvD(__A: *const c_double, __IA: isize,
                             __C: *mut c_double, __N: vDSP_Length);

    
    // vector minimum & maximum magnitudes
    pub(crate) fn vDSP_maxmgv(__A: *const c_float, __IA: isize,
                              __C: *mut c_float, __N: vDSP_Length);

    pub(crate) fn vDSP_maxmgvD(__A: *const c_double, __IA: isize,
                               __C: *mut c_double, __N: vDSP_Length);

    pub(crate) fn vDSP_minmgv(__A: *const c_float, __IA: isize,
                              __C: *mut c_float, __N: vDSP_Length);

    pub(crate) fn vDSP_minmgvD(__A: *const c_double, __IA: isize,
                               __C: *mut c_double, __N: vDSP_Length);

    
    // vector dot product
    pub(crate) fn vDSP_dotpr(__A: *const c_float, __IA: isize,
                             __B: *const c_float, __IB: isize,
                             __C: *mut c_float, __N: vDSP_Length);

    pub(crate) fn vDSP_dotprD(__A: *const c_double, __IA: isize,
                              __B: *const c_double, __IB: isize,
                              __C: *mut c_double, __N: vDSP_Length);
}
