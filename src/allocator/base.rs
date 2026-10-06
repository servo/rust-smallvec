use {
    super::Allocator,
    alloc::alloc::{
        alloc,
        dealloc,
        realloc
    },
    core::{
        alloc::Layout,
        ptr::NonNull
    }
};

#[derive(Clone)]
pub struct Global;

