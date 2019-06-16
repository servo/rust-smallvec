/// Set the length of the vec when the `SetLenOnDrop` value goes out of scope.
///
/// Copied from https://github.com/rust-lang/rust/pull/36355
pub struct SetLenOnDrop<'a> {
    len: &'a mut usize,
    local_len: usize,
}

impl<'a> SetLenOnDrop<'a> {
    #[inline]
    pub fn new(len: &'a mut usize) -> Self {
        SetLenOnDrop {
            local_len: *len,
            len: len,
        }
    }

    #[inline]
    pub fn get(&self) -> usize {
        self.local_len
    }

    #[inline]
    pub fn increment_len(&mut self, increment: usize) {
        self.local_len += increment;
    }
}

impl<'a> Drop for SetLenOnDrop<'a> {
    #[inline]
    fn drop(&mut self) {
        *self.len = self.local_len;
    }
}
