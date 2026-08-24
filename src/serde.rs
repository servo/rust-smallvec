<<<<<<< HEAD
use {
    super::SmallVec,
    core::marker::PhantomData,
    serde_core::{
        de::{SeqAccess, Visitor},
        ser::SerializeSeq,
        Deserialize, Deserializer, Serialize, Serializer,
    },
};
<<<<<<< HEAD
=======
use serde_core::{Serialize, Deserialize, Serializer, Deserializer, de::{Visitor, SeqAccess}, ser::SerializeSeq};
use super::SmallVec;
use core::marker::PhantomData;
>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)

=======
>>>>>>> 239c751 (refactor: conversions file)
impl<T, const N: usize> Serialize for SmallVec<T, N>
where
    T: Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_seq(Some(self.len()))?;
        for item in self {
            state.serialize_element(item)?;
        }
        state.end()
    }
}
impl<'de, T, const N: usize> Deserialize<'de> for SmallVec<T, N>
where
    T: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(SmallVecVisitor {
            phantom: PhantomData,
        })
    }
}
struct SmallVecVisitor<T, const N: usize> {
    phantom: PhantomData<T>,
}
impl<'de, T, const N: usize> Visitor<'de> for SmallVecVisitor<T, N>
where
    T: Deserialize<'de>,
{
    type Value = SmallVec<T, N>;
<<<<<<< HEAD
<<<<<<< HEAD
    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("a sequence")
    }
=======

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("a sequence")
    }

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
=======
    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("a sequence")
    }
>>>>>>> 239c751 (refactor: conversions file)
    fn visit_seq<B>(self, mut seq: B) -> Result<Self::Value, B::Error>
    where
        B: SeqAccess<'de>,
    {
        use serde_core::de::Error;
        let len = seq.size_hint().unwrap_or(0);
        let mut values = SmallVec::new();
        values.try_reserve(len).map_err(B::Error::custom)?;
<<<<<<< HEAD
<<<<<<< HEAD
        while let Some(value) = seq.next_element()? {
            values.push(value);
        }
        Ok(values)
    }
}
=======

=======
>>>>>>> 239c751 (refactor: conversions file)
        while let Some(value) = seq.next_element()? {
            values.push(value);
        }
        Ok(values)
    }
}
>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
