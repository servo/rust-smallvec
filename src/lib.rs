// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Small vectors in various sizes. These store a certain number of elements inline, and fall back
//! to the heap for larger allocations.  This can be a useful optimization for improving cache
//! locality and reducing allocator traffic for workloads that fit within the inline buffer.
//!
//! ## `alloc` support
//!
//! By default, `smallvec` uses the `alloc` crate, which means that it can be used on platforms
//! that have `liballoc` but not `libstd`.
//!
//! ## `union` feature
//!
//! When the `union` feature is enabled `smallvec` will track its state (inline or spilled)
//! without the use of an enum tag, reducing the size of the `smallvec` by one machine word.
//! This means that there is potentially no space overhead compared to `Vec`.
//! Note that `smallvec` can still be larger than `Vec` if the inline buffer is larger than two
//! machine words.
//!
//! To use this feature add `features = ["union"]` in the `smallvec` section of Cargo.toml.
//! Note that this feature requires a nightly compiler (for now).

#![cfg_attr(feature = "const_generics", feature(const_generics))]
#![cfg_attr(feature = "may_dangle", feature(dropck_eyepatch))]
#![cfg_attr(feature = "specialization", feature(specialization))]
#![cfg_attr(feature = "union", feature(untagged_unions))]
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

extern crate alloc;

#[cfg(not(feature = "const_generics"))]
mod array;
mod drain;
mod extend_from_slice;
mod into_iter;
#[macro_use]
mod macros;
mod set_len_on_drop;
mod small_vec;
mod small_vec_data;
mod small_vec_index_impls;
#[cfg(feature = "serde")]
mod small_vec_visitor;
#[cfg(feature = "specialization")]
mod spec_from;
#[cfg(test)]
mod tests;
mod utils;

#[cfg(not(feature = "const_generics"))]
pub use self::array::Array;
pub use self::{
    drain::Drain, extend_from_slice::ExtendFromSlice, into_iter::IntoIter, small_vec::SmallVec,
};
