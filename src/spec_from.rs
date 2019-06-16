#[cfg(not(feature = "const_generics"))]
use crate::Array;
use crate::SmallVec;

pub trait SpecFrom<A: Array, S> {
    fn spec_from(slice: S) -> SmallVec<A>;
}
