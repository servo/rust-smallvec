<<<<<<< HEAD
use {
    alloc::alloc::Layout,
    core::{
        error::Error,
        fmt::{Display, Formatter, Result as Format},
    },
};
<<<<<<< HEAD
<<<<<<< HEAD
=======
use alloc::alloc::Layout;
use core::error::Error;
use core::fmt::{Display, Formatter, Result as Format};
>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)

=======
>>>>>>> 239c751 (refactor: conversions file)
=======

>>>>>>> 716921a (fix: newline amount)
/// Error type for APIs with fallible heap allocation
#[derive(Debug)]
pub enum AllocationError {
    /// Overflow `usize::MAX` or other error during size computation
    CapacityOverflow,
    /// The allocator return an error
    Failure {
        /// The layout that was passed to the allocator
        layout: Layout,
    },
}

impl Display for AllocationError {
    fn fmt(&self, f: &mut Formatter) -> Format {
        write!(f, "Allocation error: {:?}", self)
    }
}

impl Error for AllocationError {}
