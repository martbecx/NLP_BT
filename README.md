# Backtranslation for Low-Resource Machine Translation

This repository contains scripts and utilities used to evaluate different training setups for low-resource machine translation using backtranslation (BT) strategies. All models are based on the Helsinki-NLP/OPUS-MT architecture.

## Overview

The code supports experiments for the following research questions:

1. **Fine-tuning with authentic vs. synthetic data**
2. **Effects of data selection using cross-entropy**
3. **Cumulative exposure to synthetic data (RQ3)**
4. **Staged training (incremental exposure per epoch, RQ4)**

## Scripts

### `bt_finetune.py`
Used for **Research Questions 1 and 2**.  
- Compares models trained with original, backtranslated, and combined datasets.
- Includes an option for filtering synthetic data using cross-entropy difference between in-domain and out-of-domain models.

### `cumalative_runs.py`
Used for **Research Question 3**.  
- Gradually accumulates synthetic (backtranslated) data over training epochs to observe the effect on overfitting and generalization.

### `starter_selectionepochs.py`
Used for **Research Question 4**.  
- Implements staged training, where only part of the data is exposed per epoch to simulate curriculum learning and analyze robustness to overfitting.

### `starter.yml`
Contains default HuggingFace training parameters and environment setup.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
