use crate::{Array, SmallVec};
use abomonation::Abomonation;
use std::io::Result as IOResult;
use std::io::Write;
use std::mem;

// This method currently enables undefined behavior, by exposing padding bytes.
// Copied from abomonation.
#[inline]
unsafe fn typed_to_bytes<T>(slice: &[T]) -> &[u8] {
    std::slice::from_raw_parts(
        slice.as_ptr() as *const u8,
        slice.len() * mem::size_of::<T>(),
    )
}

impl<A: Array> Abomonation for SmallVec<A>
where
    A::Item: Abomonation,
{
    #[inline]
    unsafe fn entomb<W: Write>(&self, write: &mut W) -> IOResult<()> {
        if self.spilled() {
            write.write_all(typed_to_bytes(&self[..]))?;
        }
        for element in self.iter() {
            element.entomb(write)?;
        }
        Ok(())
    }

    #[inline]
    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        // extract memory from bytes to back our smallvec
        let binary_len = if self.spilled() {
            self.len() * mem::size_of::<A::Item>()
        } else {
            0
        };
        if binary_len > bytes.len() {
            None
        } else {
            let (mine, mut rest) = bytes.split_at_mut(binary_len);
            if self.spilled() {
                let slice =
                    std::slice::from_raw_parts_mut(mine.as_mut_ptr() as *mut A::Item, self.len());
                // If the vector has spilled but then been truncated down to
                // less than the capacity, we must lie about the capacity to
                // maintain the spilled invariant.  This is ok, as the
                // exhumed smallvec is read-only.
                let capacity = Self::inline_capacity().saturating_add(1).max(self.len());
                std::ptr::write(
                    self,
                    SmallVec::from_raw_parts(slice.as_mut_ptr(), self.len(), capacity),
                );
            }
            for element in self.iter_mut() {
                let temp = rest; // temp variable explains lifetimes (mysterious!)
                rest = element.exhume(temp)?;
            }
            Some(rest)
        }
    }

    #[inline]
    fn extent(&self) -> usize {
        let mut sum = 0;
        if self.spilled() {
            sum += mem::size_of::<A::Item>() * self.len();
        }
        for element in self.iter() {
            sum += element.extent();
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use crate::SmallVec;
    use alloc::borrow::ToOwned;
    use alloc::string::String;
    use alloc::vec::Vec;

    #[test]
    fn test_abomonate_empty() {
        let v = SmallVec::<[String; 2]>::new();

        let mut bytes = Vec::new();
        unsafe {
            abomonation::encode(&v, &mut bytes).expect("encode should succeed");
        }

        if let Some((result, remaining)) =
            unsafe { abomonation::decode::<SmallVec<[String; 2]>>(&mut bytes) }
        {
            assert!(result == &v);
            assert!(remaining.len() == 0);
        }
    }

    #[test]
    fn test_abomonate_unspilled() {
        let mut v = SmallVec::<[String; 2]>::new();
        v.push("hello".to_owned());

        let mut bytes = Vec::new();
        unsafe {
            abomonation::encode(&v, &mut bytes).expect("encode should succeed");
        }

        if let Some((result, remaining)) =
            unsafe { abomonation::decode::<SmallVec<[String; 2]>>(&mut bytes) }
        {
            assert!(result == &v);
            assert!(remaining.len() == 0);
        }
    }

    #[test]
    fn test_abomonate_spilled() {
        let mut v = SmallVec::<[String; 2]>::new();
        v.push("hello".to_owned());
        v.push("there".to_owned());
        v.push("burma".to_owned());
        v.push("shave".to_owned());

        let mut bytes = Vec::new();
        unsafe {
            abomonation::encode(&v, &mut bytes).expect("encode should succeed");
        }

        if let Some((result, remaining)) =
            unsafe { abomonation::decode::<SmallVec<[String; 2]>>(&mut bytes) }
        {
            assert!(result == &v);
            assert!(remaining.len() == 0);
        }
    }

    #[test]
    fn test_abomonate_spilled_truncated() {
        let mut v = SmallVec::<[String; 2]>::new();
        v.push("hello".to_owned());
        v.push("there".to_owned());
        v.push("burma".to_owned());
        v.push("shave".to_owned());
        v.truncate(1);

        let mut bytes = Vec::new();
        unsafe {
            abomonation::encode(&v, &mut bytes).expect("encode should succeed");
        }

        if let Some((result, remaining)) =
            unsafe { abomonation::decode::<SmallVec<[String; 2]>>(&mut bytes) }
        {
            assert!(result == &v);
            assert!(result.len() == 1);
            assert!(result.capacity() == 3);
            assert!(remaining.len() == 0);
        }
    }

    #[test]
    fn test_abomonate_zst() {
        let mut v = SmallVec::<[(); 2]>::new();
        v.push(());
        v.push(());
        v.push(());
        v.push(());

        let mut bytes = Vec::new();
        unsafe {
            abomonation::encode(&v, &mut bytes).expect("encode should succeed");
        }

        if let Some((result, remaining)) =
            unsafe { abomonation::decode::<SmallVec<[(); 2]>>(&mut bytes) }
        {
            assert!(result == &v);
            assert!(result.len() == 4);
            assert!(remaining.len() == 0);
        }
    }
}
