# Detecting Outliers in Market Data: From Z-Scores to Machine Learning

Market data is the foundation of every financial decision. Yet, missing data, stale quotes, or sudden jumps can distort analysis.

In this article, we explore methods for market data anomaly detection using the FRED H.15 interest rate series.

## Step 1: Basics

- **Missing data**: find gaps in the series.
- **Stale rates**: detect when rates don't move for several days.
- **Z-scores**: flag large deviations from rolling mean.

## Step 2: Context Window Matters

Short windows flag too much noise, long windows miss local shifts. Choosing the right rolling window is key.

## Step 3: Beyond Z-scores

We compare with **Isolation Forest**, an unsupervised ML model. It captures multivariate patterns and can spot anomalies invisible to z-scores.

## Takeaways

- Always start with simple rules: missing values, stale checks.
- Z-scores provide a baseline.
- ML can add robustness, but interpretability matters.
