// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::mem::MaybeUninit;

/// A type that has the same memory layout as:
///
/// ```ignore
/// #[repr(C)]
/// union TaggedUnion2<A, B> {
///     a: Variant<A>,
///     b: Variant<B>,
///     uninit: (),
/// }
///
/// #[repr(C)]
/// struct Variant<Payload> {
///     tag: usize,
///     payload: Payload,
/// }
/// ```
///
/// … but works on Rust versions where unions fields must be `Copy`:
///
/// * https://github.com/rust-lang/rust/issues/32836
/// * https://github.com/rust-lang/rust/issues/55149
pub(crate) struct TaggedUnion2<A, B> {
    invalid_enum: MaybeUninit<InvalidEnum<A, B>>,
}

/// We rely on [RFC 2195] to construct a type that has the desired memory layout.
/// However, we’re going to store bit patterns in the enum’s tag
/// that do not correspond to any actual variant of the enum,
/// therefore violating its [validity invariant][VI].
/// To avoid undefined behavior, we never manipulate this enum type directly.
/// We store it in a `MaybeUninit`,
/// and cast pointers to it to other types before doing anything else.
///
/// [RFC 2195]: https://rust-lang.github.io/rfcs/2195-really-tagged-unions.html
/// [VI]: https://rust-lang.github.io/unsafe-code-guidelines/glossary.html#validity-and-safety-invariant
#[derive(Copy, Clone)]
#[repr(usize)]
enum InvalidEnum<PayloadA, PayloadB> {
    #[allow(unused)]
    A(PayloadA),
    #[allow(unused)]
    B(PayloadB),
}

#[repr(C)]
struct Variant<Payload> {
    tag: usize,
    payload: Payload,
}

impl<A, B> TaggedUnion2<A, B> {
    pub fn new_uninit() -> Self {
        TaggedUnion2 {
            invalid_enum: MaybeUninit::uninit(),
        }
    }

    pub fn tag(&self) -> usize {
        let ptr = self.invalid_enum.as_ptr() as *const Variant<()>;
        unsafe { (*ptr).tag }
    }

    pub fn set_tag(&mut self, tag: usize) {
        let ptr = self.invalid_enum.as_mut_ptr() as *mut Variant<()>;
        unsafe { (*ptr).tag = tag }
    }

    pub fn payload_unchecked<Payload>(&self) -> *const Payload {
        let ptr = self.invalid_enum.as_ptr() as *const Variant<Payload>;
        unsafe { &(*ptr).payload }
    }

    pub fn payload_mut_unchecked<Payload>(&mut self) -> *mut Payload {
        let ptr = self.invalid_enum.as_mut_ptr() as *mut Variant<Payload>;
        unsafe { &mut (*ptr).payload }
    }
}
