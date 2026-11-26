//! Shared support for the IMM integration-test targets.
//!
//! Each target declares `#[allow(dead_code)] mod support;` because it uses
//! only part of this code and CI denies warnings. These modules contain no
//! `#[test]` functions, so no test is compiled into more than one target.

pub mod imm_fixtures;
pub mod imm_oracle;
