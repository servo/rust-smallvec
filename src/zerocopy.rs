use {
    super::{
        SmallVec,
        allocator::global::Global
    },
    zerocopy::{
        FromZeros,
        Immutable,
        Maybe,
        TryFromBytes,
        invariant::Alignment
    }
};

#[cfg_attr(docsrs, doc(cfg(feature = "zerocopy")))]
unsafe impl<Item, const INLINE: usize> TryFromBytes for SmallVec<Item, INLINE, Global> {
    fn only_derive_is_allowed_to_implement_this_trait() {}

    #[inline]
    fn is_bit_valid<Align>(_candidate: Maybe<'_, Self, Align>) -> bool
    where Align: Alignment {
        // Disallow arbitrary byte reads; only `FromZeros::new_zeroed()` is
        // supported.
        false
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "zerocopy")))]
unsafe impl<Item, const INLINE: usize> FromZeros for SmallVec<Item, INLINE, Global> {
    fn only_derive_is_allowed_to_implement_this_trait() {}
}

// Items may be stored inline, so `Item` must be `Immutable` too.
#[cfg_attr(docsrs, doc(cfg(feature = "zerocopy")))]
unsafe impl<Item: Immutable, const INLINE: usize> Immutable for SmallVec<Item, INLINE, Global> {
    fn only_derive_is_allowed_to_implement_this_trait() {}
}
