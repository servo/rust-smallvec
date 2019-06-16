#[cfg(not(feature = "const_generics"))]
use crate::Array;
use crate::SmallVec;
use core::{fmt, marker::PhantomData};
use serde::de::{Deserialize, SeqAccess, Visitor};

macro_rules! create_with_parts {
    (
    <$($({$s_impl_ty_prefix:ident})? $s_impl_ty:ident$(: $s_impl_ty_bound:ident)?),*>,
    <$s_decl_ty:ident$(, {$s_decl_const_ty:ident})?>
) => {

#[cfg(feature = "serde")]
pub struct SmallVecVisitor<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> {
    pub(crate) phantom: PhantomData<$s_decl_ty$(, {$s_decl_const_ty})?>,
}

#[cfg(feature = "serde")]
impl<'de, $($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Visitor<'de>
    for SmallVecVisitor<$s_decl_ty$(, {$s_decl_const_ty})?>
where
    A::Item: Deserialize<'de>,
{
    type Value = SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a sequence")
    }

    fn visit_seq<B>(self, mut seq: B) -> Result<Self::Value, B::Error>
    where
        B: SeqAccess<'de>,
    {
        let len = seq.size_hint().unwrap_or(0);
        let mut values = SmallVec::with_capacity(len);

        while let Some(value) = seq.next_element()? {
            values.push(value);
        }

        Ok(values)
    }
}

    }
}

#[cfg(feature = "const_generics")]
create_with_parts!(<T, {const} N: usize>, <T, {N}>);
#[cfg(not(feature = "const_generics"))]
create_with_parts!(<A: Array>, <A>);
