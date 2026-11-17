# Impact of Spike Sorting Quality on Intracortical BCI Decoding

This repository contains the analysis code used for the project:

**Impact of Spike Sorting Quality on Intracortical BCI Decoding**

Authors: **Sai Dore and Kye Xiong**

## Project Overview

This project investigates whether spike-sorting quality affects the performance of intracortical brain-computer interface (BCI) decoding.

The analysis compares several neural feature representations and measures how well each representation can decode two-dimensional cursor velocity.

The main conditions are:

- **Condition A:** All Kilosort4 sorted units
- **Condition B:** Only sorted units that pass predefined quality criteria
- **Condition C:** Raw threshold crossings detected independently on each electrode
- **Condition D:** Random subsets of sorted units matched to the number of quality-passing units

The project also evaluates whether individual spike-sorting quality metrics are related to how much each unit contributes to decoder performance.

## Dataset

The analysis uses the publicly available nonhuman primate reaching dataset from O'Doherty et al.

Three recording sessions were analyzed:

- `indy_20160915_01`
- `indy_20160930_05`
- `indy_20161025_04`

The recordings contain neural activity from a 96-channel Utah array along with cursor and movement data.

Raw dataset files are not included in this repository because of their size.

## Analysis Pipeline

The main analysis script performs the following steps:

1. Loads behavioral and neural recording data.
2. Calculates two-dimensional cursor velocity.
3. Bins neural activity and behavior into 50 ms intervals.
4. Runs Kilosort4 spike sorting through SpikeInterface.
5. Computes spike-sorting quality metrics.
6. Filters units using predefined quality criteria.
7. Creates the four neural representation conditions.
8. Evaluates within-session decoding using a Kalman filter.
9. Performs leave-one-unit-out analysis to estimate individual unit decoding contribution.
10. Compares unit quality metrics with decoding contribution.
11. Creates electrode-level representations for cross-session decoding.
12. Trains decoders on earlier sessions and evaluates them on later sessions.

## Spike-Sorting Quality Criteria

A sorted unit is considered quality-passing when it satisfies:

- SNR ≥ 3.0
- ISI violation ratio ≤ 0.5
- Amplitude cutoff ≤ 0.10
- Amplitude CV median is available

Amplitude CV is included as a measured quality metric but is not assigned a separate hard threshold.

## Decoding

Neural activity is represented using 50 ms firing-rate bins.

The primary decoder is a Kalman filter that predicts:

- X cursor velocity
- Y cursor velocity

Performance is evaluated using:

- R² for X velocity
- R² for Y velocity
- Mean R²
- RMSE

Within-session decoding uses a chronological 80/20 training and testing split.

## Cross-Session Analysis

For cross-session decoding, sorted units are converted into a common 96-electrode representation.

Each sorted unit is assigned to the electrode with the largest absolute template waveform amplitude.

The following representations are compared across sessions:

- All sorted units pooled by electrode
- Quality-passing units pooled by electrode
- Raw threshold crossings by electrode

The decoder is trained on an earlier recording session and tested on a later recording session to evaluate decoding stability over time.

## Main File

`spike_sorting_analysis.py`

This file contains the complete within-session and cross-session analysis used for the project.

## Requirements

The analysis was developed in Python using packages including:

- NumPy
- pandas
- Matplotlib
- SciPy
- scikit-learn
- h5py
- SpikeInterface
- Kilosort4

The original analysis was performed in a Google Colab environment.

Some file paths in the script therefore use the `/content/` directory. These paths should be updated if the analysis is run in another environment.

## Data Files

To reproduce the full analysis, the original session data are required, including the corresponding MATLAB behavioral files and NWB broadband neural recordings.

Kilosort output ZIP files may also be reused to avoid rerunning spike sorting.

These large data files are not included in this repository.

## Notes

This repository is intended to provide the analysis code used for the final project and to support reproducibility of the reported methodology and results.
