# NIFTY 50 Stock Price Prediction

A machine learning-based web application for analyzing NIFTY 50 market data and generating short-term price predictions through an interactive Streamlit dashboard.

## Overview

This project uses historical NIFTY 50 market data, technical indicators, and machine learning models to provide market analysis and short-term price predictions.

The application is designed to demonstrate how financial market data can be collected, processed, analyzed, and used with machine learning models in an interactive web application.

## Features

- NIFTY 50 market data collection using Yahoo Finance
- Historical market data analysis
- Data preprocessing and validation
- Technical indicator calculation
- Moving averages
- RSI (Relative Strength Index)
- Price and volume change analysis
- Volatility analysis
- High-Low spread analysis
- Short-term price prediction
- Market trend analysis
- Risk assessment
- Model performance evaluation
- Interactive charts and visualizations
- Streamlit-based web dashboard

## Machine Learning Models

The application uses two regression models:

### Random Forest Regressor

A tree-based machine learning model used to capture nonlinear relationships between market features and the prediction target.

### Linear Regression

A statistical baseline model used to estimate the relationship between input features and the prediction target.

## Technical Indicators

The project generates several features from market data, including:

- Moving Average (5 days)
- Moving Average (10 days)
- Moving Average (20 days)
- Price Change
- Volume Change
- High-Low Spread
- Volatility
- RSI

These features are used during the machine learning workflow.

## Machine Learning Workflow

```text
NIFTY 50 Market Data
        ↓
Data Collection
        ↓
Data Cleaning & Validation
        ↓
Feature Engineering
        ↓
Technical Indicators
        ↓
Model Training
        ↓
Random Forest / Linear Regression
        ↓
Prediction
        ↓
Trend & Risk Analysis
        ↓
Streamlit Dashboard
