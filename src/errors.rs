use core::{
    alloc::Layout,
    error::Error,
    fmt::{Display, Formatter, Result as FormatResult},
};

/// Error type for APIs with fallible heap allocation
#[derive(Debug)]
pub enum CollectionAllocErr {
    /// Overflow `usize::MAX` or other error during size computation
    CapacityOverflow,
    /// The allocator return an error
    AllocErr {
        /// The layout that was passed to the allocator
        layout: Layout,
    },
}
impl Display for CollectionAllocErr {
    fn fmt(&self, f: &mut Formatter) -> FormatResult { write!(f, "Allocation error: {:?}", self) }
}

impl Error for CollectionAllocErr {}
