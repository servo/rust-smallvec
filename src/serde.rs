use {
    super::SmallVec,
    core::marker::PhantomData,
    serde_core::{
        de::{
            Deserialize,
            Deserializer,
            SeqAccess,
            Visitor
        },
        ser::{
            Serialize,
            SerializeSeq,
            Serializer
        }
    }
};

impl<T, const N: usize> Serialize for SmallVec<T, N>
where T: Serialize
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
where T: Deserialize<'de>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(SmallVecVisitor {
            phantom: PhantomData
        })
    }
}

struct SmallVecVisitor<T, const N: usize> {
    phantom: PhantomData<T>
}

impl<'de, T, const N: usize> Visitor<'de> for SmallVecVisitor<T, N>
where T: Deserialize<'de>
{
    type Value = SmallVec<T, N>;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("a sequence")
    }

    fn visit_seq<B>(self, mut seq: B) -> Result<Self::Value, B::Error>
    where B: SeqAccess<'de> {
        use serde_core::de::Error;
        let len = seq.size_hint().unwrap_or(0);
        let mut values = SmallVec::new();
        values.try_reserve(len).map_err(B::Error::custom)?;

        while let Some(value) = seq.next_element()? {
            values.push(value);
        }

        Ok(values)
    }
}
