# Diagonally Regularized Full-Matrix Calibration

## Introduction

This is the implementation of the **Diagonally Regularized Full-Matrix Calibration** introduced in the paper **"Physics-guided computational hyperspectral imaging"**.

## Input Calibration Samples
The script fits a calibration matrix that maps measured 9-channel sampling values to their corresponding theoretical sampling values. The measured values have been processed by basic pre-processing (dark-level subtraction and denoising) and channel-wise flat-field correction (FFC). For demonstration purposes, this repository directly provides the prepared calibration sample sets:

```text
calibration_samples_9ch_measured.npy
calibration_samples_9ch_theoretical.npy
```

In practical use, users may construct their own calibration sample sets from measured calibration targets and corresponding theoretical sampling values.

## Main Options

Several parameters in the script can be adjusted:

- `calibration_mode`: switches between diagonal calibration (`"diag"`) and full-matrix calibration (`"full"`).
- `full_reg_alpha`: controls the strength of the diagonal regularization in full-matrix calibration.
- `fit_indices_1based`: selects which calibration samples are used for optimization. The indices are 1-based. For example, this can be used to exclude low-signal samples such as the black patch, as done in the paper.
- `full_reg_target`: selects the regularization target. `"diag"` uses the diagonal calibration result as the prior, while `"I"` uses the identity matrix.

The fitting is performed using the original, unnormalized sampling values. Normalization is only used for visualization in the comparison figures.

## Outputs

The script saves the following outputs:

- `calib_coeff_W.npy`: calibration matrix `W`, shape `(9, 9)`, mapping measured values to the theoretical domain.
- `calib_coeff_g.npy`: channel-wise gain `g`, only saved in `diag` mode.
- `calib_coeff_meta.json`: metadata, including the calibration mode, selected fitting samples, matrix diagnostics, and evaluation metrics.
- `meas_calibrated_24x9.npy`: calibrated measured values, shape `(K, 9)`. Although the default filename contains `24x9`, the code can process any number of calibration samples.
- `compare_norm_sample_XX.png`: normalized comparison between raw measured and theoretical sampling values for sample `XX`.
- `compare_norm_sample_XX_calibrated.png`: normalized comparison between calibrated measured and theoretical sampling values for sample `XX`.

## Usage

Place the script and calibration samples in the following structure:

```text
open_version/
├── fit_calibration_matrix_9ch.py
├── calibration_set/
│   ├── calibration_samples_9ch_measured.npy
│   └── calibration_samples_9ch_theoretical.npy
└── calib_out/
```

Then run:

```bash
python fit_calibration_matrix_9ch.py
```

The output files will be saved to the folder specified by `save_dir` in the script.

## Requirements

```text
numpy
matplotlib
```

Install dependencies with:

```bash
pip install numpy matplotlib
```
