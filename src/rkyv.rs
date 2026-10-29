use {
    super::{
        SmallVec,
        allocator::{
            Allocator,
            global::Global
        }
    },
    rkyv::{
        Archive,
        Archived,
        Deserialize as RkyvDeserialize,
        Place,
        Serialize as RkyvSerialize,
        rancor::Fallible,
        ser::{
            Allocator as RkyvAllocator,
            Writer
        },
        vec::{
            ArchivedVec,
            VecResolver
        }
    }
};

impl<Item: Archive, const INLINE: usize, Heap: Allocator> Archive for SmallVec<Item, INLINE, Heap> {
    type Archived = ArchivedVec<Archived<Item>>;
    type Resolver = VecResolver;

    fn resolve(&self, resolver: Self::Resolver, out: Place<Self::Archived>) {
        ArchivedVec::resolve_from_slice(self, resolver, out);
    }
}

impl<Item, S, const INLINE: usize, Heap: Allocator> RkyvSerialize<S>
    for SmallVec<Item, INLINE, Heap>
where
    Item: RkyvSerialize<S>,
    S: Fallible + RkyvAllocator + Writer + ?Sized
{
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        ArchivedVec::serialize_from_slice(self, serializer)
    }
}

impl<Item, D, const INLINE: usize> RkyvDeserialize<SmallVec<Item, INLINE, Global>, D>
    for ArchivedVec<Archived<Item>>
where
    Item: Archive,
    Archived<Item>: RkyvDeserialize<Item, D>,
    D: Fallible + ?Sized
{
    fn deserialize(
        &self,
        deserializer: &mut D
    ) -> Result<SmallVec<Item, INLINE, Global>, D::Error> {
        let mut result = SmallVec::<Item, INLINE, Global>::with_capacity(self.len());
        for item in self.as_slice() {
            result.push(item.deserialize(deserializer)?);
        }
        Ok(result)
    }
}
