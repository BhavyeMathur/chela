#![allow(non_snake_case)]

use std::ffi::{c_double, c_float};

#[cfg(use_apple_accelerate)]
#[link(name = "Accelerate")]
extern {
    // vector sum
    pub(crate) fn vDSP_sve(__A: *const c_float, __I: isize, __C: *mut c_float, __N: isize);

    pub(crate) fn vDSP_sveD(__A: *const c_double, __I: isize, __C: *mut c_double, __N: isize);

    // vector minimum & maximum
    pub(crate) fn vDSP_maxv(__A: *const c_float, __IA: isize, __C: *mut c_float, __N: isize);

    pub(crate) fn vDSP_maxvD(__A: *const c_double, __IA: isize, __C: *mut c_double, __N: isize);

    pub(crate) fn vDSP_minv(__A: *const c_float, __IA: isize, __C: *mut c_float, __N: isize);

    pub(crate) fn vDSP_minvD(__A: *const c_double, __IA: isize, __C: *mut c_double, __N: isize);

    // vector minimum & maximum magnitudes
    pub(crate) fn vDSP_maxmgv(__A: *const c_float, __IA: isize, __C: *mut c_float, __N: isize);

    pub(crate) fn vDSP_maxmgvD(__A: *const c_double, __IA: isize, __C: *mut c_double, __N: isize);

    pub(crate) fn vDSP_minmg(__A: *const c_float, __IA: isize, __C: *mut c_float, __N: isize);

    pub(crate) fn vDSP_minmgD(__A: *const c_double, __IA: isize, __C: *mut c_double, __N: isize);
}
