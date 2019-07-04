#[cfg(feature = "serde")]
use crate::small_vec_visitor::SmallVecVisitor;
#[cfg(feature = "specialization")]
use crate::spec_from::SpecFrom;
use crate::utils::deallocate;
#[cfg(not(feature = "const_generics"))]
use crate::Array;
use crate::{
    set_len_on_drop::SetLenOnDrop, small_vec_data::SmallVecData, Drain, ExtendFromSlice, IntoIter,
};
use alloc::{vec, vec::Vec};
#[cfg(feature = "serde")]
use core::marker::PhantomData;
use core::{
    borrow::{Borrow, BorrowMut},
    cmp::{Eq, Ord, Ordering, PartialOrd},
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    hint::unreachable_unchecked,
    iter::{repeat, FromIterator},
    mem::{self, MaybeUninit},
    ops::{Deref, DerefMut, Index, IndexMut, Range, RangeFrom, RangeFull, RangeTo},
    ptr, slice,
};
#[cfg(feature = "serde")]
use serde::{
    de::{Deserialize, Deserializer},
    ser::{Serialize, SerializeSeq, Serializer},
};
#[cfg(feature = "std")]
use std::io;

macro_rules! create_with_parts {
(
    <$($({$s_impl_ty_prefix:ident})? $s_impl_ty:ident$(: $s_impl_ty_bound:ident)?),*>,
    <$s_decl_ty:ident$(, {$s_decl_const_ty:ident})?>,
    $array:ty,
    $array_item:ty,
    $array_size:expr
) => {

/// A `Vec`-like container that can store a small number of elements inline.
///
/// `SmallVec` acts like a vector, but can store a limited amount of data inline within the
/// `SmallVec` struct rather than in a separate allocation.  If the data exceeds this limit, the
/// `SmallVec` will "spill" its data onto the heap, allocating a new buffer to hold it.
///
/// The amount of data that a `SmallVec` can store inline depends on its backing store. The backing
/// store can be any type that implements the `Array` trait; usually it is a small fixed-sized
/// array.  For example a `SmallVec<[u64; 8]>` can hold up to eight 64-bit integers inline.
///
/// ## Example
///
/// ```rust
/// use smallvec::SmallVec;
/// let mut v = SmallVec::<[u8; 4]>::new(); // initialize an empty vector
///
/// // The vector can hold up to 4 items without spilling onto the heap.
/// v.extend(0..4);
/// assert_eq!(v.len(), 4);
/// assert!(!v.spilled());
///
/// // Pushing another element will force the buffer to spill:
/// v.push(4);
/// assert_eq!(v.len(), 5);
/// assert!(v.spilled());
/// ```
pub struct SmallVec<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> {
    // The capacity field is used to determine which of the storage variants is active:
    // If capacity <= $array_size then the inline variant is used and capacity holds the current length of the vector (number of elements actually in use).
    // If capacity > $array_size then the heap variant is used and capacity holds the size of the memory allocation.
    capacity: usize,
    data: SmallVecData<$s_decl_ty$(, {$s_decl_const_ty})?>,
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?> {
    /// Construct an empty vector
    #[inline]
    pub fn new() -> Self {
        SmallVec {
            capacity: 0,
            data: SmallVecData::<$s_decl_ty$(, {$s_decl_const_ty})?>::from_inline(MaybeUninit::uninit()),
        }
    }

    /// Construct an empty vector with enough capacity pre-allocated to store at least `n`
    /// elements.
    ///
    /// Will create a heap allocation only if `n` is larger than the inline capacity.
    ///
    /// ```
    /// # use smallvec::SmallVec;
    ///
    /// let v: SmallVec<[u8; 3]> = SmallVec::with_capacity(100);
    ///
    /// assert!(v.is_empty());
    /// assert!(v.capacity() >= 100);
    /// ```
    #[inline]
    pub fn with_capacity(n: usize) -> Self {
        let mut v = Self::new();
        v.reserve_exact(n);
        v
    }

    /// Construct a new `SmallVec` from a `Vec<$array_item>`.
    ///
    /// Elements will be copied to the inline buffer if vec.capacity() <= $array_size.
    ///
    /// ```rust
    /// use smallvec::SmallVec;
    ///
    /// let vec = vec![1, 2, 3, 4, 5];
    /// let small_vec: SmallVec<[_; 3]> = SmallVec::from_vec(vec);
    ///
    /// assert_eq!(&*small_vec, &[1, 2, 3, 4, 5]);
    /// ```
    #[inline]
    pub fn from_vec(mut vec: Vec<$array_item>) -> Self {
        if vec.capacity() <= $array_size {
            unsafe {
                let mut data = SmallVecData::<$s_decl_ty$(, {$s_decl_const_ty})?>::from_inline(MaybeUninit::uninit());
                let len = vec.len();
                vec.set_len(0);
                ptr::copy_nonoverlapping(vec.as_ptr(), data.inline_mut(), len);

                SmallVec {
                    capacity: len,
                    data,
                }
            }
        } else {
            let (ptr, cap, len) = (vec.as_mut_ptr(), vec.capacity(), vec.len());
            mem::forget(vec);

            SmallVec {
                capacity: cap,
                data: SmallVecData::<$s_decl_ty$(, {$s_decl_const_ty})?>::from_heap(ptr, len),
            }
        }
    }

    /// Constructs a new `SmallVec` on the stack from an `A` without
    /// copying elements.
    ///
    /// ```rust
    /// use smallvec::SmallVec;
    ///
    /// let buf = [1, 2, 3, 4, 5];
    /// let small_vec: SmallVec<_> = SmallVec::from_buf(buf);
    ///
    /// assert_eq!(&*small_vec, &[1, 2, 3, 4, 5]);
    /// ```
    #[inline]
    pub fn from_buf(buf: $array) -> Self {
        SmallVec {
            capacity: $array_size,
            data: SmallVecData::<$s_decl_ty$(, {$s_decl_const_ty})?>::from_inline(MaybeUninit::new(buf)),
        }
    }

    /// Constructs a new `SmallVec` on the stack from an `A` without
    /// copying elements. Also sets the length, which must be less or
    /// equal to the size of `buf`.
    ///
    /// ```rust
    /// use smallvec::SmallVec;
    ///
    /// let buf = [1, 2, 3, 4, 5, 0, 0, 0];
    /// let small_vec: SmallVec<_> = SmallVec::from_buf_and_len(buf, 5);
    ///
    /// assert_eq!(&*small_vec, &[1, 2, 3, 4, 5]);
    /// ```
    #[inline]
    pub fn from_buf_and_len(buf: $array, len: usize) -> Self {
        assert!(len <= $array_size);
        unsafe { Self::from_buf_and_len_unchecked(buf, len) }
    }

    /// Constructs a new `SmallVec` on the stack from an `A` without
    /// copying elements. Also sets the length. The user is responsible
    /// for ensuring that `len <= $array_size`.
    ///
    /// ```rust
    /// use smallvec::SmallVec;
    ///
    /// let buf = [1, 2, 3, 4, 5, 0, 0, 0];
    /// let small_vec: SmallVec<_> = unsafe {
    ///     SmallVec::from_buf_and_len_unchecked(buf, 5)
    /// };
    ///
    /// assert_eq!(&*small_vec, &[1, 2, 3, 4, 5]);
    /// ```
    #[inline]
    pub unsafe fn from_buf_and_len_unchecked(buf: $array, len: usize) -> Self {
        SmallVec {
            capacity: len,
            data: SmallVecData::<$s_decl_ty$(, {$s_decl_const_ty})?>::from_inline(MaybeUninit::new(buf)),
        }
    }

    /// Sets the length of a vector.
    ///
    /// This will explicitly set the size of the vector, without actually
    /// modifying its buffers, so it is up to the caller to ensure that the
    /// vector is actually the specified size.
    pub unsafe fn set_len(&mut self, new_len: usize) {
        let (_, len_ptr, _) = self.triple_mut();
        *len_ptr = new_len;
    }

    /// The maximum number of elements this vector can hold inline
    #[inline]
    pub fn inline_size(&self) -> usize {
        $array_size
    }

    /// The number of elements stored in the vector
    #[inline]
    pub fn len(&self) -> usize {
        self.triple().1
    }

    /// Returns `true` if the vector is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of items the vector can hold without reallocating
    #[inline]
    pub fn capacity(&self) -> usize {
        self.triple().2
    }

    /// Returns a tuple with (data ptr, len, capacity)
    /// Useful to get all SmallVec properties with a single check of the current storage variant.
    #[inline]
    fn triple(&self) -> (*const $array_item, usize, usize) {
        unsafe {
            if self.spilled() {
                let (ptr, len) = self.data.heap();
                (ptr, len, self.capacity)
            } else {
                (self.data.inline(), self.capacity, $array_size)
            }
        }
    }

    /// Returns a tuple with (data ptr, len ptr, capacity)
    #[inline]
    fn triple_mut(&mut self) -> (*mut $array_item, &mut usize, usize) {
        unsafe {
            if self.spilled() {
                let (ptr, len_ptr) = self.data.heap_mut();
                (ptr, len_ptr, self.capacity)
            } else {
                (
                    self.data.inline_mut(),
                    &mut self.capacity,
                    $array_size,
                )
            }
        }
    }

    /// Returns `true` if the data has spilled into a separate heap-allocated buffer.
    #[inline]
    pub fn spilled(&self) -> bool {
        self.capacity > $array_size
    }

    /// Empty the vector and return an iterator over its former contents.
    pub fn drain(&mut self) -> Drain<$array_item> {
        unsafe {
            let ptr = self.as_mut_ptr();

            let current_len = self.len();
            self.set_len(0);

            let slice = slice::from_raw_parts_mut(ptr, current_len);

            Drain {
                iter: slice.iter_mut(),
            }
        }
    }

    /// Append an item to the vector.
    #[inline]
    pub fn push(&mut self, value: $array_item) {
        unsafe {
            let (_, &mut len, cap) = self.triple_mut();
            if len == cap {
                self.reserve(1);
            }
            let (ptr, len_ptr, _) = self.triple_mut();
            *len_ptr = len + 1;
            ptr::write(ptr.add(len), value);
        }
    }

    /// Remove an item from the end of the vector and return it, or None if empty.
    #[inline]
    pub fn pop(&mut self) -> Option<$array_item> {
        unsafe {
            let (ptr, len_ptr, _) = self.triple_mut();
            if *len_ptr == 0 {
                return None;
            }
            let last_index = *len_ptr - 1;
            *len_ptr = last_index;
            Some(ptr::read(ptr.add(last_index)))
        }
    }

    /// Re-allocate to set the capacity to `max(new_cap, inline_size())`.
    ///
    /// Panics if `new_cap` is less than the vector's length.
    pub fn grow(&mut self, new_cap: usize) {
        unsafe {
            let (ptr, &mut len, cap) = self.triple_mut();
            let unspilled = !self.spilled();
            assert!(new_cap >= len);
            if new_cap <= self.inline_size() {
                if unspilled {
                    return;
                }
                self.data = SmallVecData::<$s_decl_ty$(, {$s_decl_const_ty})?>::from_inline(MaybeUninit::uninit());
                ptr::copy_nonoverlapping(ptr, self.data.inline_mut(), len);
                self.capacity = len;
            } else if new_cap != cap {
                let mut vec = Vec::with_capacity(new_cap);
                let new_alloc = vec.as_mut_ptr();
                mem::forget(vec);
                ptr::copy_nonoverlapping(ptr, new_alloc, len);
                self.data = SmallVecData::<$s_decl_ty$(, {$s_decl_const_ty})?>::from_heap(new_alloc, len);
                self.capacity = new_cap;
                if unspilled {
                    return;
                }
            } else {
                return;
            }
            deallocate(ptr, cap);
        }
    }

    /// Reserve capacity for `additional` more elements to be inserted.
    ///
    /// May reserve more space to avoid frequent reallocations.
    ///
    /// If the new capacity would overflow `usize` then it will be set to `usize::max_value()`
    /// instead. (This means that inserting `additional` new elements is not guaranteed to be
    /// possible after calling this function.)
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        // prefer triple_mut() even if triple() would work
        // so that the optimizer removes duplicated calls to it
        // from callers like insert()
        let (_, &mut len, cap) = self.triple_mut();
        if cap - len < additional {
            let new_cap = len
                .checked_add(additional)
                .and_then(usize::checked_next_power_of_two)
                .unwrap_or(usize::max_value());
            self.grow(new_cap);
        }
    }

    /// Reserve the minimum capacity for `additional` more elements to be inserted.
    ///
    /// Panics if the new capacity overflows `usize`.
    pub fn reserve_exact(&mut self, additional: usize) {
        let (_, &mut len, cap) = self.triple_mut();
        if cap - len < additional {
            match len.checked_add(additional) {
                Some(cap) => self.grow(cap),
                None => panic!("reserve_exact overflow"),
            }
        }
    }

    /// Shrink the capacity of the vector as much as possible.
    ///
    /// When possible, this will move data from an external heap buffer to the vector's inline
    /// storage.
    pub fn shrink_to_fit(&mut self) {
        if !self.spilled() {
            return;
        }
        let len = self.len();
        if self.inline_size() >= len {
            unsafe {
                let (ptr, len) = self.data.heap();
                self.data = SmallVecData::<$s_decl_ty$(, {$s_decl_const_ty})?>::from_inline(MaybeUninit::uninit());
                ptr::copy_nonoverlapping(ptr, self.data.inline_mut(), len);
                deallocate(ptr, self.capacity);
                self.capacity = len;
            }
        } else if self.capacity() > len {
            self.grow(len);
        }
    }

    /// Shorten the vector, keeping the first `len` elements and dropping the rest.
    ///
    /// If `len` is greater than or equal to the vector's current length, this has no
    /// effect.
    ///
    /// This does not re-allocate.  If you want the vector's capacity to shrink, call
    /// `shrink_to_fit` after truncating.
    pub fn truncate(&mut self, len: usize) {
        unsafe {
            let (ptr, len_ptr, _) = self.triple_mut();
            while len < *len_ptr {
                let last_index = *len_ptr - 1;
                *len_ptr = last_index;
                ptr::drop_in_place(ptr.add(last_index));
            }
        }
    }

    /// Extracts a slice containing the entire vector.
    ///
    /// Equivalent to `&s[..]`.
    pub fn as_slice(&self) -> &[$array_item] {
        self
    }

    /// Extracts a mutable slice of the entire vector.
    ///
    /// Equivalent to `&mut s[..]`.
    pub fn as_mut_slice(&mut self) -> &mut [$array_item] {
        self
    }

    /// Remove the element at position `index`, replacing it with the last element.
    ///
    /// This does not preserve ordering, but is O(1).
    ///
    /// Panics if `index` is out of bounds.
    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> $array_item {
        let len = self.len();
        self.swap(len - 1, index);
        self.pop().unwrap_or_else(|| unsafe { unreachable_unchecked() })
    }

    /// Remove all elements from the vector.
    #[inline]
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Remove and return the element at position `index`, shifting all elements after it to the
    /// left.
    ///
    /// Panics if `index` is out of bounds.
    pub fn remove(&mut self, index: usize) -> $array_item {
        unsafe {
            let (mut ptr, len_ptr, _) = self.triple_mut();
            let len = *len_ptr;
            assert!(index < len);
            *len_ptr = len - 1;
            ptr = ptr.add(index);
            let item = ptr::read(ptr);
            ptr::copy(ptr.add(1), ptr, len - index - 1);
            item
        }
    }

    /// Insert an element at position `index`, shifting all elements after it to the right.
    ///
    /// Panics if `index` is out of bounds.
    pub fn insert(&mut self, index: usize, element: $array_item) {
        self.reserve(1);

        unsafe {
            let (mut ptr, len_ptr, _) = self.triple_mut();
            let len = *len_ptr;
            assert!(index <= len);
            *len_ptr = len + 1;
            ptr = ptr.add(index);
            ptr::copy(ptr, ptr.add(1), len - index);
            ptr::write(ptr, element);
        }
    }

    /// Insert multiple elements at position `index`, shifting all following elements toward the
    /// back.
    pub fn insert_many<I: IntoIterator<Item = $array_item>>(&mut self, index: usize, iterable: I) {
        let iter = iterable.into_iter();
        if index == self.len() {
            return self.extend(iter);
        }

        let (lower_size_bound, _) = iter.size_hint();
        assert!(lower_size_bound <= core::isize::MAX as usize); // Ensure offset is indexable
        assert!(index + lower_size_bound >= index); // Protect against overflow
        self.reserve(lower_size_bound);

        unsafe {
            let old_len = self.len();
            assert!(index <= old_len);
            let mut ptr = self.as_mut_ptr().add(index);

            // Move the trailing elements.
            ptr::copy(ptr, ptr.add(lower_size_bound), old_len - index);

            // In case the iterator panics, don't double-drop the items we just copied above.
            self.set_len(index);

            let mut num_added = 0;
            for element in iter {
                let mut cur = ptr.add(num_added);
                if num_added >= lower_size_bound {
                    // Iterator provided more elements than the hint.  Move trailing items again.
                    self.reserve(1);
                    ptr = self.as_mut_ptr().add(index);
                    cur = ptr.add(num_added);
                    ptr::copy(cur, cur.add(1), old_len - index);
                }
                ptr::write(cur, element);
                num_added += 1;
            }
            if num_added < lower_size_bound {
                // Iterator provided fewer elements than the hint
                ptr::copy(
                    ptr.add(lower_size_bound),
                    ptr.add(num_added),
                    old_len - index,
                );
            }

            self.set_len(old_len + num_added);
        }
    }

    /// Convert a SmallVec to a Vec, without reallocating if the SmallVec has already spilled onto
    /// the heap.
    pub fn into_vec(self) -> Vec<$array_item> {
        if self.spilled() {
            unsafe {
                let (ptr, len) = self.data.heap();
                let v = Vec::from_raw_parts(ptr, len, self.capacity);
                mem::forget(self);
                v
            }
        } else {
            self.into_iter().collect()
        }
    }

    /// Convert the SmallVec into an `A` if possible. Otherwise return `Err(Self)`.
    ///
    /// This method returns `Err(Self)` if the SmallVec is too short (and the `A` contains uninitialized elements),
    /// or if the SmallVec is too long (and all the elements were spilled to the heap).
    pub fn into_inner(self) -> Result<$array, Self> {
        if self.spilled() || self.len() != $array_size {
            Err(self)
        } else {
            unsafe {
                let data = ptr::read(&self.data);
                mem::forget(self);
                Ok(data.into_inline())
            }
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` such that `f(&e)` returns `false`.
    /// This method operates in place and preserves the order of the retained
    /// elements.
    pub fn retain<F: FnMut(&mut $array_item) -> bool>(&mut self, mut f: F) {
        let mut del = 0;
        let len = self.len();
        for i in 0..len {
            if !f(&mut self[i]) {
                del += 1;
            } else if del > 0 {
                self.swap(i - del, i);
            }
        }
        self.truncate(len - del);
    }

    /// Removes consecutive duplicate elements.
    pub fn dedup(&mut self)
    where
        $array_item: PartialEq<$array_item>,
    {
        self.dedup_by(|a, b| a == b);
    }

    /// Removes consecutive duplicate elements using the given equality relation.
    pub fn dedup_by<F>(&mut self, mut same_bucket: F)
    where
        F: FnMut(&mut $array_item, &mut $array_item) -> bool,
    {
        // See the implementation of Vec::dedup_by in the
        // standard library for an explanation of this algorithm.
        let len = self.len();
        if len <= 1 {
            return;
        }

        let ptr = self.as_mut_ptr();
        let mut w: usize = 1;

        unsafe {
            for r in 1..len {
                let p_r = ptr.add(r);
                let p_wm1 = ptr.add(w - 1);
                if !same_bucket(&mut *p_r, &mut *p_wm1) {
                    if r != w {
                        let p_w = p_wm1.add(1);
                        mem::swap(&mut *p_r, &mut *p_w);
                    }
                    w += 1;
                }
            }
        }

        self.truncate(w);
    }

    /// Removes consecutive elements that map to the same key.
    pub fn dedup_by_key<F, K>(&mut self, mut key: F)
    where
        F: FnMut(&mut $array_item) -> K,
        K: PartialEq<K>,
    {
        self.dedup_by(|a, b| key(a) == key(b));
    }

    /// Creates a `SmallVec` directly from the raw components of another
    /// `SmallVec`.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// * `ptr` needs to have been previously allocated via `SmallVec` for its
    ///   spilled storage (at least, it's highly likely to be incorrect if it
    ///   wasn't).
    /// * `ptr`'s `$array_item` type needs to be the same size and alignment that
    ///   it was allocated with
    /// * `length` needs to be less than or equal to `capacity`.
    /// * `capacity` needs to be the capacity that the pointer was allocated
    ///   with.
    ///
    /// Violating these may cause problems like corrupting the allocator's
    /// internal data structures.
    ///
    /// Additionally, `capacity` must be greater than the amount of inline
    /// storage `A` has; that is, the new `SmallVec` must need to spill over
    /// into heap allocated storage. This condition is asserted against.
    ///
    /// The ownership of `ptr` is effectively transferred to the
    /// `SmallVec` which may then deallocate, reallocate or change the
    /// contents of memory pointed to by the pointer at will. Ensure
    /// that nothing else uses the pointer after calling this
    /// function.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::{mem, ptr};
    /// use smallvec::{smallvec, SmallVec};
    ///
    /// fn main() {
    ///     let mut v: SmallVec<[_; 1]> = smallvec![1, 2, 3];
    ///
    ///     // Pull out the important parts of `v`.
    ///     let p = v.as_mut_ptr();
    ///     let len = v.len();
    ///     let cap = v.capacity();
    ///     let spilled = v.spilled();
    ///
    ///     unsafe {
    ///         // Forget all about `v`. The heap allocation that stored the
    ///         // three values won't be deallocated.
    ///         mem::forget(v);
    ///
    ///         // Overwrite memory with [4, 5, 6].
    ///         //
    ///         // This is only safe if `spilled` is true! Otherwise, we are
    ///         // writing into the old `SmallVec`'s inline storage on the
    ///         // stack.
    ///         assert!(spilled);
    ///         for i in 0..len {
    ///             ptr::write(p.add(i), 4 + i);
    ///         }
    ///
    ///         // Put everything back together into a SmallVec with a different
    ///         // amount of inline storage, but which is still less than `cap`.
    ///         let rebuilt = SmallVec::<[_; 2]>::from_raw_parts(p, len, cap);
    ///         assert_eq!(&*rebuilt, &[4, 5, 6]);
    ///     }
    /// }
    pub unsafe fn from_raw_parts(ptr: *mut $array_item, length: usize, capacity: usize) -> Self {
        assert!(capacity > $array_size);
        SmallVec {
            capacity,
            data: SmallVecData::<$s_decl_ty$(, {$s_decl_const_ty})?>::from_heap(ptr, length),
        }
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*>
    SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
where
    $array_item: Clone,
{
    /// Resizes the vector so that its length is equal to `len`.
    ///
    /// If `len` is less than the current length, the vector simply truncated.
    ///
    /// If `len` is greater than the current length, `value` is appended to the
    /// vector until its length equals `len`.
    pub fn resize(&mut self, len: usize, value: $array_item) {
        let old_len = self.len();

        if len > old_len {
            self.extend(repeat(value).take(len - old_len));
        } else {
            self.truncate(len);
        }
    }

    /// Creates a `SmallVec` with `n` copies of `elem`.
    /// ```
    /// use smallvec::SmallVec;
    ///
    /// let v = SmallVec::<[char; 128]>::from_elem('d', 2);
    /// assert_eq!(v, SmallVec::from_buf(['d', 'd']));
    /// ```
    pub fn from_elem(elem: $array_item, n: usize) -> Self {
        if n > $array_size {
            vec![elem; n].into()
        } else {
            let mut v = Self::new();
            unsafe {
                let (ptr, len_ptr, _) = v.triple_mut();
                let mut local_len = SetLenOnDrop::new(len_ptr);

                for i in 0..n {
                    core::ptr::write(ptr.add(i), elem.clone());
                    local_len.increment_len(1);
                }
            }
            v
        }
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
where
    $array_item: Copy,
{
    /// Copy the elements from a slice into a new `SmallVec`.
    ///
    /// For slices of `Copy` types, this is more efficient than `SmallVec::from(slice)`.
    pub fn from_slice(slice: &[$array_item]) -> Self {
        let len = slice.len();
        if len <= $array_size {
            SmallVec {
                capacity: len,
                data: SmallVecData::<$s_decl_ty$(, {$s_decl_const_ty})?>::from_inline(unsafe {
                    let mut data = MaybeUninit::<$array>::uninit();
                    let slice_mut = &mut *data.as_mut_ptr();
                    ptr::copy_nonoverlapping(slice.as_ptr(), slice_mut.as_mut_ptr(), len);
                    data
                }),
            }
        } else {
            let mut b = slice.to_vec();
            let (ptr, cap) = (b.as_mut_ptr(), b.capacity());
            mem::forget(b);
            SmallVec {
                capacity: cap,
                data: SmallVecData::<$s_decl_ty$(, {$s_decl_const_ty})?>::from_heap(ptr, len),
            }
        }
    }
    /// Copy elements from a slice into the vector at position `index`, shifting any following
    /// elements toward the back.
    ///
    /// For slices of `Copy` types, this is more efficient than `insert`.
    pub fn insert_from_slice(&mut self, index: usize, slice: &[$array_item]) {
        self.reserve(slice.len());

        let len = self.len();
        assert!(index <= len);

        unsafe {
            let slice_ptr = slice.as_ptr();
            let ptr = self.as_mut_ptr().add(index);
            ptr::copy(ptr, ptr.add(slice.len()), len - index);
            ptr::copy_nonoverlapping(slice_ptr, ptr, slice.len());
            self.set_len(len + slice.len());
        }
    }

    /// Copy elements from a slice and append them to the vector.
    ///
    /// For slices of `Copy` types, this is more efficient than `extend`.
    #[inline]
    pub fn extend_from_slice(&mut self, slice: &[$array_item]) {
        let len = self.len();
        self.insert_from_slice(len, slice);
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> AsMut<[$array_item]>
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    #[inline]
    fn as_mut(&mut self) -> &mut [$array_item] {
        self
    }
}


impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> AsRef<[$array_item]>
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    #[inline]
    fn as_ref(&self) -> &[$array_item] {
        self
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Borrow<[$array_item]>
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    #[inline]
    fn borrow(&self) -> &[$array_item] {
        self
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> BorrowMut<[$array_item]>
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    #[inline]
    fn borrow_mut(&mut self) -> &mut [$array_item] {
        self
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Clone
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
where
    $array_item: Clone,
{
    fn clone(&self) -> Self {
        let mut new_vector = Self::with_capacity(self.len());
        for element in self.iter() {
            new_vector.push((*element).clone())
        }
        new_vector
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Debug
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
where
    $array_item: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Default
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Deref
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    type Target = [$array_item];
    #[inline]
    fn deref(&self) -> &[$array_item] {
        unsafe {
            let (ptr, len, _) = self.triple();
            slice::from_raw_parts(ptr, len)
        }
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> DerefMut
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    #[inline]
    fn deref_mut(&mut self) -> &mut [$array_item] {
        unsafe {
            let (ptr, &mut len, _) = self.triple_mut();
            slice::from_raw_parts_mut(ptr, len)
        }
    }
}

#[cfg(feature = "may_dangle")]
unsafe impl<#[may_dangle] $($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Drop
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    fn drop(&mut self) {
        unsafe {
            if self.spilled() {
                let (ptr, len) = self.data.heap();
                Vec::from_raw_parts(ptr, len, self.capacity);
            } else {
                ptr::drop_in_place(&mut self[..]);
            }
        }
    }
}

#[cfg(not(feature = "may_dangle"))]
impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Drop
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    fn drop(&mut self) {
        unsafe {
            if self.spilled() {
                let (ptr, len) = self.data.heap();
                Vec::from_raw_parts(ptr, len, self.capacity);
            } else {
                ptr::drop_in_place(&mut self[..]);
            }
        }
    }
}

#[cfg(feature = "serde")]
impl<'de, $($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Deserialize<'de>
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
where
    $array_item: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(SmallVecVisitor {
            phantom: PhantomData,
        })
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Eq
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
where
    $array_item: Eq {}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Extend<$array_item>
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    fn extend<I: IntoIterator<Item = $array_item>>(&mut self, iterable: I) {
        let mut iter = iterable.into_iter();
        let (lower_size_bound, _) = iter.size_hint();
        self.reserve(lower_size_bound);

        unsafe {
            let (ptr, len_ptr, cap) = self.triple_mut();
            let mut len = SetLenOnDrop::new(len_ptr);
            while len.get() < cap {
                if let Some(out) = iter.next() {
                    ptr::write(ptr.add(len.get()), out);
                    len.increment_len(1);
                } else {
                    return;
                }
            }
        }

        for elem in iter {
            self.push(elem);
        }
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> ExtendFromSlice<$array_item>
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
where
    $array_item: Copy,
{
    fn extend_from_slice(&mut self, other: &[$array_item]) {
        Self::extend_from_slice(self, other)
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> From<$array>
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    #[inline]
    fn from(array: $array) -> Self {
        Self::from_buf(array)
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> From<Vec<$array_item>>
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    #[inline]
    fn from(vec: Vec<$array_item>) -> Self {
        Self::from_vec(vec)
    }
}

impl<'a, $($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> From<&'a [$array_item]>
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
where
    $array_item: Clone,
{
    #[cfg(not(feature = "specialization"))]
    #[inline]
    fn from(slice: &'a [$array_item]) -> Self {
        slice.into_iter().cloned().collect()
    }

    #[cfg(feature = "specialization")]
    #[inline]
    fn from(slice: &'a [$array_item]) -> Self {
        Self::spec_from(slice)
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> FromIterator<$array_item>
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    fn from_iter<I: IntoIterator<Item = $array_item>>(iterable: I) -> Self {
        let mut v = Self::new();
        v.extend(iterable);
        v
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Hash
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
where
    $array_item: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> IntoIterator
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    type IntoIter = IntoIter<$s_decl_ty$(, {$s_decl_const_ty})?>;
    type Item = $array_item;
    fn into_iter(mut self) -> Self::IntoIter {
        unsafe {
            // Set SmallVec len to zero as `IntoIter` drop handles dropping of the elements
            let len = self.len();
            self.set_len(0);
            IntoIter {
                data: self,
                current: 0,
                end: len,
            }
        }
    }
}

impl<'a, $($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> IntoIterator
    for &'a SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    type IntoIter = slice::Iter<'a, $array_item>;
    type Item = &'a $array_item;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, $($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> IntoIterator
    for &'a mut SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    type IntoIter = slice::IterMut<'a, $array_item>;
    type Item = &'a mut $array_item;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Ord
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
where
    $array_item: Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> PartialOrd
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
where
    $array_item: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

unsafe impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Send
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
where
    $array_item: Send {}


#[cfg(feature = "serde")]
impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Serialize
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
where
    $array_item: Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_seq(Some(self.len()))?;
        for item in self {
            state.serialize_element(&item)?;
        }
        state.end()
    }
}

#[cfg(feature = "specialization")]
impl<'a, $($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*>
    SpecFrom<&'a [$array_item], $s_decl_ty$(, {$s_decl_const_ty})?>
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
where
    $array_item: Clone,
{
    #[inline]
    default fn spec_from(slice: &'a [$array_item]) -> Self {
        slice.into_iter().cloned().collect()
    }
}

#[cfg(feature = "specialization")]
impl<'a, $($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*>
    SpecFrom<&'a [$array_item], $s_decl_ty$(, {$s_decl_const_ty})?>
    for SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>
where
    $array_item: Copy,
{
    #[inline]
    fn spec_from(slice: &'a [$array_item]) -> Self {
        Self::from_slice(slice)
    }
}

    }
}

#[cfg(feature = "const_generics")]
impl<T, const N: usize> PartialEq<Self> for SmallVec<T, { N }>
where
    T: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self[..] == other[..]
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        self[..] != other[..]
    }
}

#[cfg(not(feature = "const_generics"))]
impl<A: Array, B: Array> PartialEq<SmallVec<B>> for SmallVec<A>
where
    A::Item: PartialEq<B::Item>,
{
    #[inline]
    fn eq(&self, other: &SmallVec<B>) -> bool {
        self[..] == other[..]
    }
}

#[cfg(all(feature = "std", feature = "const_generics"))]
impl<const N: usize> io::Write for SmallVec<u8, { N }> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.extend_from_slice(buf);
        Ok(buf.len())
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.extend_from_slice(buf);
        Ok(())
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(all(feature = "std", not(feature = "const_generics")))]
impl<A: Array<Item = u8>> io::Write for SmallVec<A> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.extend_from_slice(buf);
        Ok(buf.len())
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.extend_from_slice(buf);
        Ok(())
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(not(feature = "const_generics"))]
macro_rules! impl_index {
    ($index_type: ty, $output_type: ty) => {
        impl<A: Array> Index<$index_type> for SmallVec<A> {
            type Output = $output_type;
            #[inline]
            fn index(&self, index: $index_type) -> &$output_type {
                &(&**self)[index]
            }
        }

        impl<A: Array> IndexMut<$index_type> for SmallVec<A> {
            #[inline]
            fn index_mut(&mut self, index: $index_type) -> &mut $output_type {
                &mut (&mut **self)[index]
            }
        }
    };
}

#[cfg(not(feature = "const_generics"))]
impl_index!(usize, A::Item);
#[cfg(not(feature = "const_generics"))]
impl_index!(Range<usize>, [A::Item]);
#[cfg(not(feature = "const_generics"))]
impl_index!(RangeFrom<usize>, [A::Item]);
#[cfg(not(feature = "const_generics"))]
impl_index!(RangeTo<usize>, [A::Item]);
#[cfg(not(feature = "const_generics"))]
impl_index!(RangeFull, [A::Item]);

#[cfg(feature = "const_generics")]
create_with_parts!(<T, {const} N: usize>, <T, {N}>, [T; N], T, N);
#[cfg(not(feature = "const_generics"))]
create_with_parts!(<A: Array>, <A>, A, A::Item, A::size());
