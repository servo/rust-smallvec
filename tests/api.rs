use smallvec::{CollectionAllocErr, SmallVec};

fn exported_alloc_error(error: CollectionAllocErr) -> CollectionAllocErr {
    error
}

#[test]
fn fallible_allocation_error_is_exported() {
    let mut vec = SmallVec::<u8, 4>::new();

    let error = vec.try_reserve(usize::MAX).unwrap_err();

    assert!(matches!(
        exported_alloc_error(error),
        CollectionAllocErr::CapacityOverflow
    ));
}
