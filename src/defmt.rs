use defmt::Format;
use defmt::Formatter;
use defmt::write;
use super::SmallVec;

impl<T: Format, const N: usize> Format for SmallVec<T, N> {
    fn format(&self, fmt: Formatter) {
        write!(fmt, "[");
        for (index, element) in self.iter().enumerate() {
            if index != 0 {write!(fmt, ", ")}
            write!(fmt, "{:?}", element);
        }
        write!(fmt, "]");
    }
}