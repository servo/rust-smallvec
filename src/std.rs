extern crate std;
<<<<<<< HEAD
<<<<<<< HEAD

<<<<<<< HEAD
use {super::SmallVec, std::io};
=======
use super::SmallVec;
use std::io;
>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)

=======
use {super::SmallVec, std::io};
>>>>>>> 239c751 (refactor: conversions file)
=======

use {super::SmallVec, std::io};

>>>>>>> 716921a (fix: newline amount)
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<const N: usize> io::Write for SmallVec<u8, N> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.extend_from_slice(buf);
        Ok(buf.len())
    }
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
=======
>>>>>>> 239c751 (refactor: conversions file)
    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.extend_from_slice(buf);
        Ok(())
    }
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
=======
>>>>>>> 239c751 (refactor: conversions file)
    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
<<<<<<< HEAD
<<<<<<< HEAD
}
=======
}
>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
=======
}
>>>>>>> 716921a (fix: newline amount)
