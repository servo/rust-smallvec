// Licensed under the Apache License, Version 0.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Cold assertion helpers for SmallVec methods.
//!
//! These functions are marked `#[cold]` and `#[inline(never)]` to improve
//! cache locality — the panic paths are not on the hot path and should not
//! pollute the instruction cache of the calling method.

/// Panics because a slice range `start..end` is out of bounds.
///
/// `start > end` means the range is inverted; `end > len` means it
/// extends past the slice.
#[cold]
#[inline(never)]
#[track_caller]
pub(crate) fn slice_range_failed(start: usize, end: usize, len: usize) -> ! {
    if start > end {
        panic!("slice index starts at {start} but ends at {end}");
    } else {
        panic!("range end index {end} out of range for slice of length {len}");
    }
}

/// Panics because `index` is out of bounds for a `SmallVec` of length `len`.
///
/// `method` is the name of the calling context, included in the panic
/// message to aid debugging.
#[cold]
#[inline(never)]
#[track_caller]
pub(crate) fn index_out_of_bounds(method: &str, index: usize, len: usize) -> ! {
    panic!("{method} index (is {index}) should be <= len (is {len})");
}
