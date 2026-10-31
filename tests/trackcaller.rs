use {
    smallvec::SmallVec,
    std::{
        cell::RefCell,
        panic::{
            AssertUnwindSafe,
            catch_unwind,
            set_hook,
            take_hook
        }
    }
};

thread_local! {
    static LOCATION: RefCell<Option<(String, u32)>> = const { RefCell::new(None) };
}

/// Runs `f` and returns the file and line its panic was reported at.
fn panic_location(f: impl FnOnce()) -> Option<(String, u32)> {
    let previous = take_hook();
    set_hook(Box::new(|info| {
        let location = info
            .location()
            .map(|location| (location.file().to_owned(), location.line()));
        LOCATION.with(|cell| *cell.borrow_mut() = location);
    }));
    let result = catch_unwind(AssertUnwindSafe(f));
    set_hook(previous);
    assert!(result.is_err(), "expected a panic");
    LOCATION.with(|cell| cell.borrow_mut().take())
}

/// Asserts that the expression panics and that the panic points at this line.
macro_rules! assert_panics_here {
    ($expr:expr) => {
        assert_eq!(
            panic_location(|| _ = $expr),
            Some((file!().to_owned(), line!())),
            "`{}` panicked at the wrong location",
            stringify!($expr)
        )
    };
}

#[test]
fn panics_point_to_caller() {
    let mut v: SmallVec<i32, 2> = SmallVec::from([1, 2, 3]);

    assert_panics_here!(v.swap_remove(10));
    assert_panics_here!(v.remove(10));
    assert_panics_here!(v.insert(10, 0));
    assert_panics_here!(v.insert_mut(10, 0));
    assert_panics_here!(v.split_off(10));
    assert_panics_here!(v.drain(2..10));
    assert_panics_here!(v.drain(..=usize::MAX));
    assert_panics_here!(v.splice(2..10, [0]));
    assert_panics_here!(v.reserve(usize::MAX));
}
