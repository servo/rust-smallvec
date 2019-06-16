/// `panic!()` in debug builds, optimization hint in release.
#[cfg(not(feature = "union"))]
macro_rules! debug_unreachable {
    () => {
        debug_unreachable!("entered unreachable code")
    };
    ($e:expr) => {
        if cfg!(not(debug_assertions)) {
            crate::utils::unreachable();
        } else {
            panic!($e);
        }
    };
}

/// Creates a [`SmallVec`] containing the arguments.
///
/// `smallvec!` allows `SmallVec`s to be defined with the same syntax as array expressions.
/// There are two forms of this macro:
///
/// - Create a [`SmallVec`] containing a given list of elements:
///
/// ```
/// # #[macro_use] extern crate smallvec;
/// # use smallvec::SmallVec;
/// # fn main() {
/// let v: SmallVec<[_; 128]> = smallvec![1, 2, 3];
/// assert_eq!(v[0], 1);
/// assert_eq!(v[1], 2);
/// assert_eq!(v[2], 3);
/// # }
/// ```
///
/// - Create a [`SmallVec`] from a given element and size:
///
/// ```
/// # #[macro_use] extern crate smallvec;
/// # use smallvec::SmallVec;
/// # fn main() {
/// let v: SmallVec<[_; 0x8000]> = smallvec![1; 3];
/// assert_eq!(v, SmallVec::from_buf([1, 1, 1]));
/// # }
/// ```
///
/// Note that unlike array expressions this syntax supports all elements
/// which implement [`Clone`] and the number of elements doesn't have to be
/// a constant.
///
/// This will use `clone` to duplicate an expression, so one should be careful
/// using this with types having a nonstandard `Clone` implementation. For
/// example, `smallvec![Rc::new(1); 5]` will create a vector of five references
/// to the same boxed integer value, not five references pointing to independently
/// boxed integers.
#[macro_export]
macro_rules! smallvec {
    // count helper: transform any expression into 1
    (@one $x:expr) => (1usize);
    ($elem:expr; $n:expr) => ({
        $crate::SmallVec::from_elem($elem, $n)
    });
    ($($x:expr),*$(,)*) => ({
        let count = 0usize $(+ smallvec!(@one $x))*;
        let mut vec = $crate::SmallVec::new();
        if count <= vec.inline_size() {
            $(vec.push($x);)*
            vec
        } else {
            $crate::SmallVec::from_vec(vec![$($x,)*])
        }
    });
}
