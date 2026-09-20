use {
    alloc::alloc::handle_alloc_error,
    core::{
        alloc::Layout,
        error::Error,
        fmt::{
            Debug,
            Display,
            Formatter,
            Result as Format
        }
    }
};

#[derive(Debug)]
pub enum SmallVecError {
    CapacityOverflow,
    AllocationError(Layout)
}

impl Display for SmallVecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Format {
        write!(f, "Allocation error: {self:?}")
    }
}

impl Error for SmallVecError {}

impl SmallVecError {
    pub fn handle<Type>(self) -> Type {
        match self {
            SmallVecError::CapacityOverflow => panic!("smallvec capacity overflow"),
            SmallVecError::AllocationError(layout) => handle_alloc_error(layout)
        }
    }
}