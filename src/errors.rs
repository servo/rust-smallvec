use {
    alloc::alloc::handle_alloc_error,
    core::{
        alloc::Layout,
        convert::Infallible,
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

pub trait Handle {
    type Handled;
    fn handle(self) -> Self::Handled;
}

impl Handle for SmallVecError {
    type Handled = Infallible;

    #[cold]
    #[inline(never)]
    fn handle(self) -> Self::Handled {
        match self {
            SmallVecError::CapacityOverflow => panic!("smallvec capacity overflow"),
            SmallVecError::AllocationError(layout) => handle_alloc_error(layout)
        }
    }
}

impl<Type> Handle for Result<Type, SmallVecError> {
    type Handled = Type;

    #[inline]
    fn handle(self) -> Self::Handled {
        match self {
            Ok(value) => value,
            Err(error) => match error.handle() {}
        }
    }
}
