use core::alloc::Layout;

/// Error type for APIs with fallible heap allocation
#[derive(Debug)]
pub enum CollectionAllocErr {
    /// Overflow `usize::MAX` or other error during size computation
    CapacityOverflow,
    /// The allocator return an error
    AllocErr {
        /// The layout that was passed to the allocator
        layout: Layout
    }
}
impl core::fmt::Display for CollectionAllocErr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Allocation error: {:?}", self)
    }
}

impl core::error::Error for CollectionAllocErr {}
