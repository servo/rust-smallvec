#[cfg(feature = "allocator-api2")]
use allocator_api2::alloc::Global;
#[cfg(not(feature = "allocator-api2"))]
use std::alloc::Global;
use smallvec::SmallVec;

#[test]
fn round_trip_with_allocator() {
    // INLINE = 2: len 2 stays inline, len 5 spills
    for len in [2usize, 5] {
        let mut v: SmallVec<String, 2, Global> = SmallVec::with_capacity_in(0, Global);
        for i in 0..len {
            v.push(i.to_string());
        }
        assert_eq!(v.spilled(), len > 2);

        let (raw, length, allocator) = v.into_parts_with_allocator();
        assert_eq!(length.len(), len);
        assert_eq!(length.on_heap(), len > 2);

        // SAFETY: parts come straight from `into_parts_with_allocator`.
        let rebuilt = unsafe {
            SmallVec::<String, 2, Global>::from_parts_with_allocator(raw, length, allocator)
        };
        assert_eq!(rebuilt.len(), len);
        assert_eq!(rebuilt[0], "0");
    }
}
