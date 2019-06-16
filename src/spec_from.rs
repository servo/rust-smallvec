#[cfg(not(feature = "const_generics"))]
use crate::Array;
use crate::SmallVec;

macro_rules! create_with_parts {
    (
    <$($({$s_impl_ty_prefix:ident})? $s_impl_ty:ident$(: $s_impl_ty_bound:ident)?),*>,
    <$s_decl_ty:ident$(, {$s_decl_const_ty:ident})?>
) => {

pub trait SpecFrom<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*, S> {
    fn spec_from(slice: S) -> SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>;
}

    }
}

#[cfg(feature = "const_generics")]
create_with_parts!(<T, {const} N: usize>);
#[cfg(not(feature = "const_generics"))]
create_with_parts!(<A: Array>, <A>);
