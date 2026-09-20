use {
    core::{
        alloc::Layout,
        error::Error,
        fmt::{
            Debug,
            Display,
            Formatter,
            Result as Format
        }
    },
    alloc::alloc::handle_alloc_error
};

pub trait Handle {
    type Handled;
    fn handle(self) -> Self::Handled;
}

#[derive(Debug)]
pub enum SmallVecError {
    CapacityOverflow,
    AllocationError(Layout)
}

impl Error for SmallVecError {}

impl Display for SmallVecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Format {
        write!(f, "Allocation error: {self:?}")
    }
}

impl<Type> Handle for Result<Type, SmallVecError> {
    type Handled = Type;
    fn handle(self) -> Self::Handled {
        match self {
            Ok(value) => value,
            Err(SmallVecError::CapacityOverflow) => panic!("smallvec capacity overflow"),
            Err(SmallVecError::AllocationError(layout)) => handle_alloc_error(layout)
        }
    }
}