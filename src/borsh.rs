use {
    super::SmallVec,
    alloc::{
        collections::BTreeMap as Map,
        format
    },
    borsh::{
        BorshSchema,
        BorshSerialize,
        io::{
            Result as Serial,
            Write
        },
        schema::{
            Declaration,
            Definition
        }
    }
};

impl<Type: BorshSerialize, const INLINE: usize> BorshSerialize for SmallVec<Type, INLINE> {
    fn serialize<Writer: Write>(&self, writer: &mut Writer) -> Serial<()> {
        self.len.0.serialize(writer)?;
        for element in self {
            element.serialize(writer)?;
        }
        return Ok(());
    }
}

impl<Type: BorshSchema, const INLINE: usize> BorshSchema for SmallVec<Type, INLINE> {
    fn declaration() -> Declaration {
        return format!("SmallVec<{}, {INLINE}>", Type::declaration());
    }

    fn add_definitions_recursively(definitions: &mut Map<Declaration, Definition>) -> () {
        let declaration = Self::declaration();
        if definitions.contains_key(&declaration) {
            return;
        }
        Type::add_definitions_recursively(definitions);
        definitions.insert(
            declaration,
            Definition::Sequence {
                length_width: usize::BITS as u8 / 8,
                length_range: 0..=(isize::MAX as u64),
                elements: Type::declaration()
            }
        );
    }
}
