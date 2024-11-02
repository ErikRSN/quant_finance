# Unsupervised Learning Trading Strategy

A Python-based trading strategy that uses K-Means clustering and portfolio optimization to select stocks from the S&P 500 index.

## Overview

This project implements a systematic trading strategy that:
1. Processes S&P 500 stocks data
2. Calculates technical indicators and features
3. Uses unsupervised learning to group similar stocks
4. Optimizes portfolio allocation using the Efficient Frontier

## Features

- Automated data collection from Yahoo Finance
- Technical indicators calculation including:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - ATR (Average True Range)
  - Garman-Klass Volatility
- Data aggregation and liquidity filtering
- Factor analysis using Fama-French model
- K-Means clustering for stock selection
- Portfolio optimization using Sharpe ratio

## Requirements

```python
pandas
numpy
pandas_ta
yfinance
statsmodels
matplotlib