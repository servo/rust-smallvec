use std::{
    env,
    path::PathBuf,
    process::Command
};

fn main() {
    println!("cargo:rerun-if-changed=syntax_macro.rs");

    // Compile the bundled macro directly so it does not need to be published
    // as a separate Cargo package.
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let status = Command::new(env::var_os("RUSTC").expect("Cargo must set RUSTC"))
        .current_dir(&manifest_dir)
        .args([
            "--crate-name=syntax",
            "--crate-type=proc-macro",
            "--edition=2024",
            "--extern=proc_macro",
            "syntax_macro.rs",
            "--out-dir"
        ])
        .arg(&out_dir)
        .status()
        .expect("failed to run rustc for the syntax proc-macro");

    assert!(status.success(), "failed to compile the syntax proc-macro");
    println!("cargo:rustc-link-search=all={}", out_dir.display());
}
