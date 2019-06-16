#[cfg(not(feature = "const_generics"))]
use crate::Array;
use crate::SmallVec;
use core::{fmt, marker::PhantomData};
use serde::de::{Deserialize, SeqAccess, Visitor};

#[cfg(feature = "serde")]
pub struct SmallVecVisitor<A> {
    pub(crate) phantom: PhantomData<A>,
}

#[cfg(feature = "serde")]
impl<'de, A: Array> Visitor<'de> for SmallVecVisitor<A>
where
    A::Item: Deserialize<'de>,
{
    type Value = SmallVec<A>;

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
