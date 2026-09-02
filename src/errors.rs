// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Error type for APIs with fallible heap allocation
#[derive(Debug)]
pub enum CollectionAllocErr {
    /// Overflow `usize::MAX` or other error during size computation
    CapacityOverflow,
    /// The allocator return an error
    AllocErr {
        /// The layout that was passed to the allocator
        layout: alloc::alloc::Layout,
    },
}

impl core::fmt::Display for CollectionAllocErr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Allocation error: {:?} ", self)
    }
}

impl core::error::Error for CollectionAllocErr {}
