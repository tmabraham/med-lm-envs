# REDACTED-eval

`REDACTED-eval` is a command-line tool for evaluating language models on medical benchmarks. It handles the full pipeline: running benchmarks, processing results, and computing model comparisons.

> **Note:** `REDACTED-eval <ENV>` and `REDACTED-eval bench` are wrappers around the [verifiers](https://github.com/primeintellect-ai/verifiers) `vf-eval` command, adding medical-specific environments, batch orchestration, and environment-specific CLI flags inferred from each benchmark's `load_environment()` signature.

## Quick Start

```bash
# Run a single benchmark (fastest way to test a model)
REDACTED-eval medqa -m gpt-4.1-mini -n 25

# Run a batch of benchmarks from a config file
REDACTED-eval bench --config configs/job-gpt-oss-20b.yaml

# Process raw results into analysis-ready parquet files
REDACTED-eval process

# Compute win rates across models
REDACTED-eval winrate
```

## Typical Workflow

```
1. Run Benchmarks  -->  2. Process Results  -->  3. Compute Win Rates
   (bench or single)       (process)               (winrate)
        |                      |                        |
        v                      v                        v
    runs/raw/           runs/processed/          runs/winrate/
```

## Commands

| Command | Purpose |
|---------|---------|
| `REDACTED-eval <ENV>` | Run a single benchmark interactively |
| `REDACTED-eval bench` | Run multiple benchmarks from a config file |
| `REDACTED-eval process` | Convert raw results to parquet for analysis |
| `REDACTED-eval winrate` | Compute model comparisons from processed data |

## Command Structure

```bash
# Single-run mode: environment name comes FIRST
REDACTED-eval medqa -m gpt-4.1-mini -n 50

# Subcommands: keyword comes first
REDACTED-eval bench --config configs/my-run.yaml
REDACTED-eval process --runs-dir runs/raw
REDACTED-eval winrate --processed-dir runs/processed
```

> **Important:** In single-run mode, the environment name must be the first argument.
> `REDACTED-eval --verbose medqa` will fail; use `REDACTED-eval medqa --verbose` instead.

## When to Use Each Mode

### Single-Run Mode (`REDACTED-eval <ENV>`)

**Best for:** Quick tests, debugging, exploring a benchmark's behavior.

```bash
# Test GPT-4.1-mini on 25 MedQA questions
REDACTED-eval medqa -m gpt-4.1-mini -n 25

# See what options are available for an environment
REDACTED-eval longhealth --help
```

### Batch Mode (`REDACTED-eval bench`)

**Best for:** Systematic evaluation across multiple models and benchmarks.

```bash
# Run all jobs defined in config
REDACTED-eval bench --config configs/job-gpt-oss-20b.yaml

# Preview what would run without executing
REDACTED-eval bench --config configs/job-gpt-oss-20b.yaml --dry-run

# Force all jobs to use a specific API endpoint
REDACTED-eval bench --config configs/job-gpt-oss-20b.yaml --api-base-url http://127.0.0.1:8000/v1
```

### Processing Mode (`REDACTED-eval process`)

**Best for:** Preparing results for analysis after batch runs complete.

```bash
# Process all completed runs
REDACTED-eval process

# Process specific directory
REDACTED-eval process --runs-dir runs/raw --output-dir runs/processed
```

### Win Rate Mode (`REDACTED-eval winrate`)

**Best for:** Comparing model performance across benchmarks.

```bash
# Compute win rates from processed results
REDACTED-eval winrate

# List available models before computing
REDACTED-eval winrate --list-models
```

## Output Directory Structure

```
runs/
├── raw/                          # Raw benchmark outputs (from bench/single-run)
│   └── <run_id>/
│       ├── run_manifest.json     # Run metadata and job status
│       └── <job_id>/             # Per-job results
│           ├── results.jsonl
│           └── summary.json
├── processed/                    # Analysis-ready parquet files (from process)
│   ├── env_index.json            # Dataset inventory
│   └── <env>/<model>.parquet
└── winrate/                      # Model comparison outputs (from winrate)
    ├── latest.json
    └── latest.csv
```

## Getting Help

```bash
REDACTED-eval --help              # General usage
REDACTED-eval bench --help        # Batch mode options
REDACTED-eval process --help      # Processing options
REDACTED-eval winrate --help      # Win rate options
REDACTED-eval medqa --help        # Environment-specific options
```

## Installing Environments

Before running benchmarks, environments must be installed. There are two ways to install environments:

### From Local Directory

Use `vf-install` for environments in the local `environments/` directory:

```bash
# Install a local environment
vf-install medqa

# Install from a specific path
vf-install ./environments/medqa
```

### From Environments Hub

Use `prime env install` to install environments from the [Prime Intellect Environments Hub](https://app.primeintellect.ai):

```bash
# Install latest version
prime env install owner/environment-name

# Install specific version
prime env install owner/environment-name@0.1.3
```

## Detailed Documentation

- [Single-Run Mode](REDACTED-eval-single-run.md) - Run individual benchmarks with custom options
- [Batch Mode](REDACTED-eval-bench.md) - Configure and run systematic evaluations
- [Processing](REDACTED-eval-process.md) - Prepare results for analysis
- [Win Rates](REDACTED-eval-winrate.md) - Compare models across benchmarks
