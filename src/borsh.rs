use {
    super::SmallVec,
    alloc::{
        collections::BTreeMap as Map,
        format
    },
    borsh::{
        BorshDeserialize,
        BorshSchema,
        BorshSerialize,
        io::{
            Error,
            ErrorKind,
            Result as Serial,
            Write
        },
        schema::{
            Declaration,
            Definition
        }
    },
    core::iter::repeat_with
};

impl<Type: BorshSerialize, const INLINE: usize> BorshSerialize for SmallVec<Type, INLINE> {
    fn serialize<Writer: Write>(&self, writer: &mut Writer) -> Serial<()> {
        (self.len.value() as u64).serialize(writer)?;
        for element in self {
            element.serialize(writer)?;
        }

        Ok(())
    }
}

impl<Type: BorshDeserialize, const INLINE: usize> BorshDeserialize for SmallVec<Type, INLINE> {
    fn deserialize_reader<R: borsh::io::Read>(reader: &mut R) -> Serial<Self> {
        let length = u64::deserialize_reader(reader)?;
        repeat_with(|| Type::deserialize_reader(reader))
            .take(length.try_into().map_err(|_| Error::new(
                ErrorKind::OutOfMemory,
                "Cannot deserialize a sequence with more than usize::MAX elements in this machine"
            ))?)
            .collect()
    }
}

impl<Type: BorshSchema, const INLINE: usize> BorshSchema for SmallVec<Type, INLINE> {
    fn declaration() -> Declaration {
        format!("Vec<{}>", Type::declaration())
    }

    fn add_definitions_recursively(definitions: &mut Map<Declaration, Definition>) {
        let declaration = Self::declaration();
        if definitions.contains_key(&declaration) {
            return;
        }
        Type::add_definitions_recursively(definitions);
        definitions.insert(
            declaration,
            Definition::Sequence {
                length_width: 8,
                length_range: 0..=u64::MAX,
                elements: Type::declaration()
            }
        );
    }
}
