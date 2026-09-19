use core::{
    alloc::Layout,
    convert::Infallible,
    error::Error,
    fmt::{
        Debug,
        Display,
        Formatter,
        Result as Format
    }
};

#[derive(Debug)]
pub struct CapacityOverflow;

impl Handle for CapacityOverflow {
    type Handled = Infallible;

    #[inline]
    fn handle(self) -> Self::Handled {
        panic!("capacity overflow")
    }
}

impl Display for CapacityOverflow {
    fn fmt(&self, f: &mut Formatter<'_>) -> Format {
        write!(f, "Allocation error: {:?}", self)
    }
}

impl Error for CapacityOverflow {}

#[derive(Debug)]
pub struct AllocationError(pub Layout);

impl Handle for AllocationError {
    type Handled = Infallible;

    #[inline]
    fn handle(self) -> Self::Handled {
        alloc::alloc::handle_alloc_error(self.0)
    }
}

impl Display for AllocationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Format {
        write!(f, "Allocation error: {:?}", self)
    }
}

impl Error for AllocationError {}

pub trait Handle {
    type Handled;
    fn handle(self) -> Self::Handled;
}

impl<Type, Do: Handle<Handled = Infallible>> Handle for Result<Type, Do> {
    type Handled = Type;

    #[inline]
    fn handle(self) -> Self::Handled {
        match self {
            Ok(value) => value,
            #[allow(unreachable_code)]
            Err(error) => match error.handle() {}
        }
    }
}

#[derive(Debug)]
pub enum SmallVecError {
    CapacityOverflow(CapacityOverflow),
    AllocationError(AllocationError)
}

impl Handle for SmallVecError {
    type Handled = Infallible;

    #[inline]
    fn handle(self) -> Self::Handled {
        match self {
            Self::CapacityOverflow(error) => error.handle(),
            Self::AllocationError(error) => error.handle()
        }
    }
}

impl Display for SmallVecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Format {
        write!(f, "Allocation error: {:?}", self)
    }
}

impl Error for SmallVecError {}
