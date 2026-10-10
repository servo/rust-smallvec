use {
    super::{Allocator, Global, SmallVec},
    core::marker::PhantomData,
    serde_core::{
        de::{Deserialize, Deserializer, SeqAccess, Visitor},
        ser::{Serialize, SerializeSeq, Serializer},
    },
};

impl<Item: Serialize, const INLINE: usize, Heap: Allocator> Serialize
    for SmallVec<Item, INLINE, Heap>
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_seq(Some(self.len()))?;
        for item in self {
            state.serialize_element(item)?;
        }
        state.end()
    }
}

impl<'de, Item: Deserialize<'de>, const INLINE: usize> Deserialize<'de>
    for SmallVec<Item, INLINE, Global>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(SmallVecVisitor {
            phantom: PhantomData,
        })
    }
}

struct SmallVecVisitor<Item, const INLINE: usize> {
    phantom: PhantomData<Item>,
}

impl<'de, Item: Deserialize<'de>, const INLINE: usize> Visitor<'de>
    for SmallVecVisitor<Item, INLINE>
{
    type Value = SmallVec<Item, INLINE, Global>;

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("a sequence")
    }

    fn visit_seq<B: SeqAccess<'de>>(self, mut seq: B) -> Result<Self::Value, B::Error> {
        use serde_core::de::Error;
        let length = seq.size_hint().unwrap_or(0);
        let mut values = SmallVec::new();
        values.try_reserve(length).map_err(B::Error::custom)?;

        while let Some(value) = seq.next_element()? {
            values.push(value);
        }

        Ok(values)
    }
}
