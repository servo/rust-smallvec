use core::mem;

pub unsafe fn deallocate<T>(ptr: *mut T, capacity: usize) {
    let _vec: Vec<T> = Vec::from_raw_parts(ptr, 0, capacity);
    // Let it drop.
}

/// Hint to the optimizer that any code path which calls this function is
/// statically unreachable and can be removed.
///
/// Equivalent to `std::hint::unreachable_unchecked` but works in older versions of Rust.
#[inline]
pub unsafe fn unreachable() -> ! {
    enum Void {}
    let x: &Void = mem::transmute(1usize);
    match *x {}
}
