#[deprecated(since = "2.0.0-alpha.13", note = "use `SmallVec::from` instead")]
#[macro_export]
macro_rules! smallvec {
    ($elem:expr; $n:expr) => ({
        $crate::from_elem($elem, $n)
    });
    ($($($x:expr),+$(,)?)?) => ({
        $crate::SmallVec::from([$($($x),+)?])
    });
}

#[deprecated(since = "2.0.0-alpha.13", note = "use `SmallVec::from_buf` instead")]
#[macro_export]
macro_rules! smallvec_inline {
    ($elem:expr; $n:expr) => ({
        $crate::SmallVec::<_, $n, $crate::Global>::from_buf([$elem; $n])
    });
    ($($($x:expr),+$(,)?)?) => ({
        $crate::SmallVec::from_buf([$($($x),+)?])
    });
}

macro_rules! public {
    (#[cfg($($cond:tt)*)] $(#[$attr:meta])* const fn $($rest:tt)*) => {
        #[cfg($($cond)*)]
        $(#[$attr])*
        pub const fn $($rest)*

        #[cfg(not($($cond)*))]
        $(#[$attr])*
        const fn $($rest)*
    };
    (#[cfg($($cond:tt)*)] $(#[$attr:meta])* fn $($rest:tt)*) => {
        #[cfg($($cond)*)]
        $(#[$attr])*
        pub fn $($rest)*

        #[cfg(not($($cond)*))]
        $(#[$attr])*
        fn $($rest)*
    };
}
