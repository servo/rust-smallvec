use super::SmallVec;
use defmt::write;
use defmt::Format;
use defmt::Formatter;

impl<T: Format, const N: usize> Format for SmallVec<T, N> {
    fn format(&self, fmt: Formatter) {
        write!(fmt, "[");
        for (index, element) in self.iter().enumerate() {
            if index != 0 {
                write!(fmt, ", ")
            }
            write!(fmt, "{:?}", element);
        }
        write!(fmt, "]");
    }
}
