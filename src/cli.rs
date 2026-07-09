use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use rsomics_common::{CommonFlags, Result, RsomicsError, ToolMeta};
use rsomics_help::{Example, FlagSpec, HelpSpec, Origin, Section};

use rsomics_mantel::{Alternative, MantelResult, Method, mantel, read_matrix, write_result};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

#[derive(Parser, Debug)]
#[command(name = "rsomics-mantel", version, about, long_about = None, disable_help_flag = true)]
pub struct Cli {
    /// First distance matrix (lsmat TSV: blank corner, id header, then rows).
    dm1: PathBuf,

    /// Second distance matrix; reordered onto the first's ids.
    dm2: PathBuf,

    #[arg(short = 'm', long, default_value = "pearson")]
    method: String,

    #[arg(short = 'p', long, default_value_t = 999)]
    permutations: usize,

    #[arg(short = 'a', long, default_value = "two-sided")]
    alternative: String,

    #[arg(short = 'o', long, default_value = "-")]
    output: String,

    #[command(flatten)]
    pub common: CommonFlags,
}

impl Cli {
    /// Run the Mantel test and, unless `--json` is set, write the result
    /// table to the chosen output. Under `--json` the framework serialises
    /// the returned struct into the result envelope, so nothing is written
    /// to stdout here.
    pub fn report(self) -> Result<MantelResult> {
        self.common.install_rayon_pool()?;
        let method = Method::parse(&self.method)?;
        let alternative = Alternative::parse(&self.alternative)?;
        let seed = self.common.seed_rng();

        let x = read_matrix(open(&self.dm1)?, &self.dm1.display().to_string())?;
        let y = read_matrix(open(&self.dm2)?, &self.dm2.display().to_string())?;
        if x.n() != y.n() {
            return Err(RsomicsError::InvalidInput(format!(
                "matrices differ in size: {} vs {}",
                x.n(),
                y.n()
            )));
        }
        let y_data = y.reorder_like(&x.ids, &self.dm2.display().to_string())?;

        let res = mantel(
            &x.data,
            &y_data,
            x.n(),
            method,
            self.permutations,
            alternative,
            seed,
        );

        if !self.common.json {
            let mut out: Box<dyn Write> = if self.output == "-" {
                Box::new(BufWriter::new(std::io::stdout().lock()))
            } else {
                Box::new(BufWriter::new(
                    File::create(&self.output).map_err(RsomicsError::Io)?,
                ))
            };
            write_result(&mut out, &res)?;
            out.flush().map_err(RsomicsError::Io)?;
        }

        if !self.common.quiet {
            eprintln!(
                "{} test: r={:.6}, p={:.4}, n={}",
                method.name(),
                res.r,
                res.p_value,
                res.n
            );
        }
        Ok(res)
    }
}

fn open(path: &std::path::Path) -> Result<BufReader<File>> {
    File::open(path)
        .map(BufReader::new)
        .map_err(|e| RsomicsError::InvalidInput(format!("{}: {e}", path.display())))
}

pub static HELP: HelpSpec = HelpSpec {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
    tagline: "Mantel test: permutation correlation between two distance matrices.",
    origin: Some(Origin {
        upstream: "scikit-bio skbio.stats.distance.mantel",
        upstream_license: "BSD-3-Clause",
        our_license: "MIT OR Apache-2.0",
        paper_doi: Some("PMID:6018555"),
    }),
    usage_lines: &["<dm1.tsv> <dm2.tsv> [-m pearson] [-p 999] [-a two-sided] [-o out.tsv]"],
    sections: &[Section {
        title: "OPTIONS",
        flags: &[
            FlagSpec {
                short: Some('m'),
                long: "method",
                aliases: &[],
                value: Some("<pearson|spearman>"),
                type_hint: Some("String"),
                required: false,
                default: Some("pearson"),
                description: "Correlation method; spearman ranks the distances first.",
                why_default: None,
            },
            FlagSpec {
                short: Some('p'),
                long: "permutations",
                aliases: &[],
                value: Some("<int>"),
                type_hint: Some("usize"),
                required: false,
                default: Some("999"),
                description: "Permutations for the p-value; 0 skips it (p = nan).",
                why_default: None,
            },
            FlagSpec {
                short: Some('a'),
                long: "alternative",
                aliases: &[],
                value: Some("<two-sided|greater|less>"),
                type_hint: Some("String"),
                required: false,
                default: Some("two-sided"),
                description: "Alternative hypothesis for the permutation p-value.",
                why_default: None,
            },
            FlagSpec {
                short: Some('o'),
                long: "output",
                aliases: &[],
                value: Some("<path>"),
                type_hint: Some("String"),
                required: false,
                default: Some("-"),
                description: "Output path (- for stdout).",
                why_default: None,
            },
        ],
    }],
    examples: &[
        Example {
            description: "Pearson Mantel with 999 permutations",
            command: "rsomics-mantel a.tsv b.tsv -o result.tsv",
        },
        Example {
            description: "Spearman, one-sided greater, fixed seed",
            command: "rsomics-mantel a.tsv b.tsv -m spearman -a greater --seed 42",
        },
    ],
    json_result_schema_doc: None,
};

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_debug_assert() {
        Cli::command().debug_assert();
    }
}
