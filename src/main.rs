mod cli;

use clap::Parser;
use cli::{Cli, HELP, META};
use rsomics_help::{intercept_help, render as render_help};
use std::process::ExitCode;

fn main() -> ExitCode {
    let raw_args: Vec<String> = std::env::args().collect();
    if let Some(mode) = intercept_help(&raw_args) {
        render_help(&HELP, mode);
        return ExitCode::SUCCESS;
    }
    let cli = Cli::parse();
    let common = cli.common.clone();
    rsomics_common::runner::run(&common, META, || cli.report())
}
