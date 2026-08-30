use {
    super::SmallVec,
    borsh::{
        BorshSerialize,
        io::{
            Result as Serial,
            Write
        }
    }
};

impl<Type: BorshSerialize, const INLINE: usize> BorshSerialize for SmallVec<Type, INLINE> {
    fn serialize<Writer: Write>(&self, writer: &mut Writer) -> Serial<()> {
        usize::from(self.len).serialize(writer)?;
        for element in self {
            element.serialize(writer)?;
        }
        return Ok(());
    }
}
