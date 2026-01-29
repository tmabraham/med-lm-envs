# Processing Results

Convert raw benchmark outputs into analysis-ready parquet files. This step prepares data for win rate computation and other analyses.

## Quick Start

```bash
# Process all completed runs (uses defaults)
REDACTED-eval process

# Specify directories explicitly
REDACTED-eval process --runs-dir runs/raw --output-dir runs/processed

# Preview what would be processed
REDACTED-eval process --dry-run
```

## What Processing Does

1. **Discovers** completed jobs in `runs/raw/`
2. **Extracts** results from each job's output files
3. **Normalizes** data into a consistent schema
4. **Writes** parquet files organized by environment and model
5. **Creates** an index (`env_index.json`) for downstream tools

### Output Structure

```
runs/processed/
├── env_index.json              # Dataset inventory for winrate/analysis
├── medqa/
│   ├── gpt-4o.parquet
│   └── gpt-4o-mini.parquet
├── pubmedqa/
│   ├── gpt-4o.parquet
│   └── gpt-4o-mini.parquet
└── ...
```

## Common Options

| Flag | Description | Default |
|------|-------------|---------|
| `--runs-dir PATH` | Directory containing raw runs | `runs/raw` |
| `--output-dir PATH` | Where to write processed files | `runs/processed` |
| `--max-workers N` | Parallel processing threads | 4 |
| `--dry-run` | Show what would be processed | - |
| `--yes` | Skip confirmation prompts | - |

## Filtering Runs

### By Completion Status

By default, only completed jobs are processed:

```bash
# Include incomplete runs
REDACTED-eval process --process-incomplete

# Filter by specific status
REDACTED-eval process --status completed --status failed
```

### Latest Runs Only

When multiple runs exist for the same (model, environment) pair, processing uses the latest by default.

## Clean Rebuild

Delete all processed outputs and rebuild from scratch:

```bash
# Interactive confirmation
REDACTED-eval process --clean

# Non-interactive (for scripts)
REDACTED-eval process --clean --yes
```

## Using a Config File

Store common options in a YAML file:

```yaml
# process-config.yaml
runs_dir: runs/raw
output_dir: runs/processed
max_workers: 8
process_incomplete: false
```

```bash
REDACTED-eval process --config process-config.yaml
```

CLI flags override config values.

## Hugging Face Integration

Sync processed datasets to/from the Hugging Face Hub:

```yaml
# process-config.yaml
runs_dir: runs/raw
output_dir: runs/processed

hf:
  repo: your-org/medical-benchmarks
  branch: main
  token: ${HF_TOKEN}
  private: true
```

### Pull Before Processing

```bash
# Prompt before pulling
REDACTED-eval process --hf-repo your-org/data --hf-pull-policy prompt

# Always pull existing data first
REDACTED-eval process --hf-repo your-org/data --hf-pull-policy pull

# Start fresh (ignore remote)
REDACTED-eval process --hf-repo your-org/data --hf-pull-policy clean
```

### Push After Processing

When `--hf-repo` is set, processed files are automatically uploaded after completion.

## Chaining with Win Rates

Process and compute win rates in one step:

```bash
REDACTED-eval process --winrate winrate-config.yaml
```

This runs `REDACTED-eval winrate` automatically after processing completes.

## Example Workflows

### Basic Processing Pipeline

```bash
# 1. Run benchmarks
REDACTED-eval bench --config my-eval.yaml

# 2. Process results
REDACTED-eval process

# 3. Compute win rates
REDACTED-eval winrate
```

### CI/CD Pipeline

```bash
# Non-interactive processing with cleanup
REDACTED-eval process \
  --runs-dir ./benchmark-outputs \
  --output-dir ./processed \
  --clean \
  --yes \
  --max-workers 16
```

### Incremental Updates

```bash
# Process only new runs (default behavior)
REDACTED-eval process

# env_index.json tracks what's already processed
```

## Troubleshooting

### "No runs found"

Check that:
1. `--runs-dir` points to the correct location
2. Runs have completed (check `run_manifest.json` status)
3. Use `--process-incomplete` if runs are still in progress

### Missing data in output

By default, only jobs with `completed` status are included. Use `--process-incomplete` to include partial results.

## Next Steps

After processing, [compute win rates](REDACTED-eval-winrate.md) to compare model performance.
