//! CLI: `thresh-rcs-compute` — compute a monostatic RCS sweep for an STL
//! target via the PyPOFacets bridge and write the result to JSON.
//!
//! Usage: see `thresh_synth::rcs_compute::cli::usage_text()` or run
//! `thresh-rcs-compute --help`.
//!
//! Only built with the `rcs-compute` feature because it links against the
//! PyO3 Python bridge used by `RcsComputeBridge`. Argument parsing itself
//! lives in `thresh_synth::rcs_compute::cli` and is unit tested there
//! without pulling in the Python runtime.

use std::process::ExitCode;

use thresh_synth::rcs_compute::{cli, compute_and_save_rcs};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match cli::parse_args(&args) {
        Ok(cli::ParsedArgs::Help) => {
            eprintln!("{}", cli::usage_text());
            ExitCode::SUCCESS
        }
        Ok(cli::ParsedArgs::Run {
            stl,
            output,
            config,
        }) => {
            let sample_count = cli::config_sample_count(&config);
            match compute_and_save_rcs(&stl, &config, &output) {
                Ok(()) => {
                    eprintln!("wrote {} samples to {}", sample_count, output.display());
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("thresh-rcs-compute: RCS computation failed: {err}");
                    ExitCode::from(2)
                }
            }
        }
        Err(err) => {
            eprintln!("thresh-rcs-compute: {err}");
            eprintln!();
            eprintln!("{}", cli::usage_text());
            ExitCode::from(1)
        }
    }
}
