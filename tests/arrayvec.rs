use {
    arrayvec::ArrayVec,
    smallvec::SmallVec,
    std::{
        cell::Cell,
        cmp::Ordering,
        rc::Rc
    }
};

/// Counts how many times it has been dropped, to catch leaks and double drops.
struct Tracked(Rc<Cell<usize>>);

impl Drop for Tracked {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

fn tracked<const N: usize>(count: usize, drops: &Rc<Cell<usize>>) -> ArrayVec<Tracked, N> {
    (0..count).map(|_| Tracked(drops.clone())).collect()
}

#[test]
fn from_arrayvec_stays_inline_or_spills() {
    let small: SmallVec<i32, 4> = ArrayVec::<i32, 8>::from_iter([1, 2, 3]).into();
    assert!(!small.spilled());
    assert_eq!(&small[..], &[1, 2, 3]);

    let small: SmallVec<i32, 2> = ArrayVec::<i32, 8>::from_iter([1, 2, 3, 4, 5]).into();
    assert!(small.spilled());
    assert_eq!(&small[..], &[1, 2, 3, 4, 5]);
}

#[test]
fn try_from_smallvec_fits_or_returns_original() {
    // Inline source, exactly filling the array.
    let array = ArrayVec::<i32, 4>::try_from(SmallVec::<i32, 4>::from([1, 2, 3, 4])).unwrap();
    assert_eq!(&array[..], &[1, 2, 3, 4]);
    // Spilled source.
    let array = ArrayVec::<i32, 8>::try_from(SmallVec::<i32, 2>::from([1, 2, 3, 4, 5])).unwrap();
    assert_eq!(&array[..], &[1, 2, 3, 4, 5]);

    // One element too many: the original comes back untouched.
    let error = ArrayVec::<i32, 3>::try_from(SmallVec::<i32, 4>::from([1, 2, 3, 4])).unwrap_err();
    assert!(!error.spilled());
    assert_eq!(&error[..], &[1, 2, 3, 4]);
    let error =
        ArrayVec::<i32, 4>::try_from(SmallVec::<i32, 2>::from([1, 2, 3, 4, 5])).unwrap_err();
    assert!(error.spilled());
    assert_eq!(&error[..], &[1, 2, 3, 4, 5]);
}

#[test]
fn conversions_drop_each_element_exactly_once() {
    // `From`, both inline (3 <= 4) and spilled (6 > 4).
    for count in [3, 6] {
        let drops = Rc::new(Cell::new(0));
        let small: SmallVec<Tracked, 4> = tracked::<8>(count, &drops).into();
        assert_eq!(small.spilled(), count > 4);
        assert_eq!(drops.get(), 0, "conversion must not drop elements");
        drop(small);
        assert_eq!(drops.get(), count);
    }

    // `TryFrom` success, both inline and spilled.
    for count in [3, 6] {
        let drops = Rc::new(Cell::new(0));
        let small: SmallVec<Tracked, 4> = tracked::<8>(count, &drops).into();
        let array = ArrayVec::<Tracked, 8>::try_from(small).ok().unwrap();
        assert_eq!(drops.get(), 0, "conversion must not drop elements");
        drop(array);
        assert_eq!(drops.get(), count);
    }

    // `TryFrom` failure hands every element back.
    let drops = Rc::new(Cell::new(0));
    let small: SmallVec<Tracked, 2> = tracked::<8>(5, &drops).into();
    let error = ArrayVec::<Tracked, 3>::try_from(small).err().unwrap();
    assert_eq!(drops.get(), 0, "failed conversion must not drop elements");
    drop(error);
    assert_eq!(drops.get(), 5);
}

#[test]
fn zero_sized_items() {
    let array: ArrayVec<(), 8> = std::iter::repeat_n((), 5).collect();
    let small: SmallVec<(), 2> = array.into();
    assert_eq!(small.len(), 5);

    assert!(ArrayVec::<(), 4>::try_from(small.clone()).is_err());
    assert_eq!(ArrayVec::<(), 5>::try_from(small).unwrap().len(), 5);
}

#[test]
fn eq_across_capacities_and_item_types() {
    let small = SmallVec::<i32, 2>::from([1, 2, 3]);
    let array = ArrayVec::<i32, 8>::from_iter([1, 2, 3]);
    assert!(small == array);
    assert!(array == small);
    assert!(small != ArrayVec::<i32, 8>::from_iter([1, 2, 4]));
    assert!(array != SmallVec::<i32, 2>::from([1, 2]));

    // `Item: PartialEq<Other>` with different types.
    let small: SmallVec<String, 2> = SmallVec::from(["a".to_string(), "b".to_string()]);
    let array = ArrayVec::<&str, 4>::from_iter(["a", "b"]);
    assert!(small == array);
    assert!(array == small);
}

#[test]
fn ordering_in_both_directions() {
    let small = SmallVec::<i32, 2>::from([1, 2, 3]);
    let array = ArrayVec::<i32, 8>::from_iter([1, 2, 4]);
    assert!(small < array);
    assert!(array > small);
    assert_eq!(small.partial_cmp(&array), Some(Ordering::Less));
    assert_eq!(array.partial_cmp(&small), Some(Ordering::Greater));
}
