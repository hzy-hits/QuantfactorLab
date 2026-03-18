/// Fast bootstrap significance test for factor IC.
///
/// Usage:
///   echo "0.03 -0.01 0.02 0.05 -0.02 ..." | rust-bootstrap --n 100000
///   → {"real_ic": 0.026, "p_value": 0.0001, "significant": true}
///
/// Input: daily IC values (one per line or space-separated)
/// Output: JSON with p-value
use std::io::{self, Read};

use rand::prelude::*;
use rayon::prelude::*;
use serde::Serialize;

#[derive(Serialize)]
struct BootstrapResult {
    real_ic: f64,
    p_value: f64,
    null_mean: f64,
    null_std: f64,
    percentile: f64,
    significant_01: bool,
    significant_05: bool,
    n_bootstrap: usize,
    n_days: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_bootstrap: usize = args
        .iter()
        .position(|a| a == "--n")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    // Read daily ICs from stdin
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();

    let daily_ics: Vec<f64> = input
        .split_whitespace()
        .filter_map(|s| s.parse::<f64>().ok())
        .collect();

    if daily_ics.is_empty() {
        eprintln!("No daily IC values provided");
        std::process::exit(1);
    }

    let n_days = daily_ics.len();
    let real_ic: f64 = daily_ics.iter().sum::<f64>() / n_days as f64;
    let abs_real_ic = real_ic.abs();

    // Parallel bootstrap with rayon
    let null_ics: Vec<f64> = (0..n_bootstrap)
        .into_par_iter()
        .map(|seed| {
            let mut rng = StdRng::seed_from_u64(42 + seed as u64);
            let mut sum = 0.0f64;
            for &ic in &daily_ics {
                // Random sign flip under null hypothesis (IC mean = 0)
                let sign: f64 = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
                // Resample with replacement
                let idx = rng.gen_range(0..n_days);
                sum += daily_ics[idx] * sign;
            }
            sum / n_days as f64
        })
        .collect();

    // p-value: two-sided
    let n_extreme = null_ics.iter().filter(|&&x| x.abs() >= abs_real_ic).count();
    let p_value = n_extreme as f64 / n_bootstrap as f64;

    let null_mean: f64 = null_ics.iter().sum::<f64>() / n_bootstrap as f64;
    let null_var: f64 = null_ics.iter().map(|x| (x - null_mean).powi(2)).sum::<f64>() / n_bootstrap as f64;
    let null_std = null_var.sqrt();

    let n_below = null_ics.iter().filter(|&&x| x < real_ic).count();
    let percentile = n_below as f64 / n_bootstrap as f64 * 100.0;

    let result = BootstrapResult {
        real_ic: (real_ic * 1_000_000.0).round() / 1_000_000.0,
        p_value: (p_value * 1_000_000.0).round() / 1_000_000.0,
        null_mean: (null_mean * 1_000_000.0).round() / 1_000_000.0,
        null_std: (null_std * 1_000_000.0).round() / 1_000_000.0,
        percentile: (percentile * 10.0).round() / 10.0,
        significant_01: p_value < 0.01,
        significant_05: p_value < 0.05,
        n_bootstrap,
        n_days,
    };

    println!("{}", serde_json::to_string(&result).unwrap());
}
