use {
    super::{Allocator, Global, SmallVec},
    alloc::{collections::BTreeMap as Map, format},
    borsh::{
        BorshDeserialize, BorshSchema, BorshSerialize,
        io::{Error, ErrorKind, Result as Serial, Write},
        schema::{Declaration, Definition},
    },
    core::iter::repeat_with,
};

impl<Item: BorshSerialize, const INLINE: usize, Heap: Allocator> BorshSerialize
    for SmallVec<Item, INLINE, Heap>
{
    fn serialize<Writer: Write>(&self, writer: &mut Writer) -> Serial<()> {
        (self.len() as u64).serialize(writer)?;
        for element in self {
            element.serialize(writer)?;
        }

        Ok(())
    }
}

impl<Item: BorshDeserialize, const INLINE: usize> BorshDeserialize
    for SmallVec<Item, INLINE, Global>
{
    fn deserialize_reader<R: borsh::io::Read>(reader: &mut R) -> Serial<Self> {
        let length = u64::deserialize_reader(reader)?;
        repeat_with(|| Item::deserialize_reader(reader))
            .take(length.try_into().map_err(|_| Error::new(
                ErrorKind::OutOfMemory,
                "Cannot deserialize a sequence with more than usize::MAX elements in this machine"
            ))?)
            .collect()
    }
}

impl<Item: BorshSchema, const INLINE: usize, Heap: Allocator> BorshSchema
    for SmallVec<Item, INLINE, Heap>
{
    fn declaration() -> Declaration {
        format!("Vec<{}>", Item::declaration())
    }

    fn add_definitions_recursively(definitions: &mut Map<Declaration, Definition>) {
        let declaration = Self::declaration();
        if definitions.contains_key(&declaration) {
            return;
        }
        Item::add_definitions_recursively(definitions);
        definitions.insert(
            declaration,
            Definition::Sequence {
                length_width: 8,
                length_range: 0..=u64::MAX,
                elements: Item::declaration(),
            },
        );
    }
}
