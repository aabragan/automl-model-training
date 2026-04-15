# Sample Datasets

Ready-to-run CSV files for each supported use case.

| File | Use Case | Label | Command |
|------|----------|-------|---------|
| `fraud_detection.csv` | Binary classification | `is_fraud` | `uv run train-binary samples/fraud_detection.csv --label is_fraud` |
| `flower_species.csv` | Multiclass classification | `species` | `uv run train samples/flower_species.csv --label species` |
| `house_prices.csv` | Regression | `price` | `uv run train-regression samples/house_prices.csv --label price` |
| `monthly_sales.csv` | Temporal backtest | `sales` | `uv run backtest samples/monthly_sales.csv --date-column date --label sales --n-splits 3` |

## Quickstart

```bash
# Binary classification — fraud detection
uv run train-binary samples/fraud_detection.csv --label is_fraud --time-limit 60

# Regression — house price prediction
uv run train-regression samples/house_prices.csv --label price --time-limit 60

# Multiclass — flower species
uv run train samples/flower_species.csv --label species --time-limit 60

# Temporal backtest — monthly sales
uv run backtest samples/monthly_sales.csv --date-column date --label sales --n-splits 3
```

## macOS note

The default `best` preset uses LightGBM heavily. If you see `Library not loaded: libomp.dylib`, install the missing dependency:

```bash
brew install libomp
```

Alternatively, use `--preset medium_quality` to avoid the issue entirely:

```bash
uv run train-binary samples/fraud_detection.csv --label is_fraud --time-limit 60 --preset medium_quality
```
