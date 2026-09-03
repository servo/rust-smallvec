// By default, `defmt` will log error-level messages and no other level.
// Wether a message at a given level is emitted is decided at compile time,
// so this level is chosen to avoid making this test special inside CI.
use {
    defmt::error,
    defmt2log::init_from_current_exe,
    log::{
        Level,
        LevelFilter,
        Metadata,
        Record
    },
    smallvec::SmallVec,
    std::sync::Mutex
};

// Logging infrastructure taken from the `defmt2log` crate test suite.

struct TestLogger {
    records: Mutex<SmallVec<(Level, String), 16>>
}

impl TestLogger {
    fn install(&'static self) {
        if log::set_logger(self).is_ok() {
            log::set_max_level(LevelFilter::Trace);
        }
        self.records.lock().unwrap().clear();
    }

    fn take(&self) -> Vec<(Level, String)> {
        self.records.lock().unwrap().drain(..).collect()
    }
}

impl log::Log for TestLogger {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &Record<'_>) {
        self.records
            .lock()
            .unwrap()
            .push((record.level(), record.args().to_string()));
    }

    fn flush(&self) {}
}

static LOGGER: TestLogger = TestLogger {
    records: Mutex::new(SmallVec::new())
};

#[test]
#[cfg_attr(miri, ignore = "isolated environment makes execution impossible")]
#[cfg_attr(target_os = "windows", ignore = "windows is not supported by `defmt`")]
fn test_defmt() {
    init_from_current_exe();
    LOGGER.install();

    let mut expected = SmallVec::<&'static str, 16>::new();

    // Vectors storing bytes.

    // Empty vector.
    let s: SmallVec<u8, 2> = SmallVec::new();
    error!("{=[?]}", s);
    expected.push("[]");

    // Vector with inline storage.
    let s: SmallVec<u8, 2> = SmallVec::from([1, 4, 9, 16]);
    error!("{=[?]}", s);
    expected.push("[1, 4, 9, 16]");

    // Vector with heap storage.
    let s: SmallVec<u8, 8> = SmallVec::from([2, 5, 10, 17]);
    error!("{=[?]}", s);
    expected.push("[2, 5, 10, 17]");

    // Vectors storing string slices.

    // Empty vector.
    let s: SmallVec<&'static str, 2> = SmallVec::new();
    error!("{=[?]}", s);
    expected.push("[]");

    // Vector with inline storage.
    let s: SmallVec<&'static str, 2> = SmallVec::from(["a", "b", "c", "d"]);
    error!("{=[?]}", s);
    expected.push("[a, b, c, d]");

    // Vector with heap storage.
    let s: SmallVec<&'static str, 8> = SmallVec::from(["e", "f", "g", "h"]);
    error!("{=[?]}", s);
    expected.push("[e, f, g, h]");

    let records = LOGGER.take();
    for (actual, expected) in records.into_iter().map(|x| x.1).zip(expected.into_iter()) {
        assert_eq!(expected, actual);
    }
}
