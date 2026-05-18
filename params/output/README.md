# params/output

This folder contains the original MATLAB scripts for registration evaluation and comparison, plus Python equivalents that replicate the same computations and plotting behavior.

## Files

- `compute_errors.m` / `compute_errors.py`
  - Computes registration success rate and error metrics from affine result files.
- `registration_results.m` / `registration_results.py`
  - Generates the small-scale delta yaw plot from hardcoded success rate vectors.
- `registration_results_full.m` / `registration_results_full.py`
  - Generates the full set of plots from the MATLAB script with hardcoded result vectors.
- `soybean_row3_cpd_comparison.m` / `soybean_row3_cpd_comparison.py`
- `soybean_row4_cpd_comparison.m` / `soybean_row4_cpd_comparison.py`
- `soybean_row5_cpd_comparison.m` / `soybean_row5_cpd_comparison.py`
  - Compute CPD comparison metrics and export results to CSV and NPY.
- `output_utils.py`
  - Shared helper functions used by the Python scripts.

## Python usage

Install dependencies:

```bash
python3 -m pip install numpy matplotlib
```

Run a script from `params/output`:

```bash
cd /home/f/tesina/AgriColMap/params/output
python3 compute_errors.py
python3 registration_results.py
python3 registration_results_full.py
python3 soybean_row3_cpd_comparison.py
python3 soybean_row4_cpd_comparison.py
python3 soybean_row5_cpd_comparison.py
```

## Output

- Plot files created by the Python scripts are written to `params/output/plots/`.
- CPD comparison scripts also save their numeric results as:
  - `soybean_row3_cpd_comparison_result.csv`
  - `soybean_row3_cpd_comparison_result.npy`
  - `soybean_row4_cpd_comparison_result.csv`
  - `soybean_row4_cpd_comparison_result.npy`
  - `soybean_row5_cpd_comparison_result.csv`
  - `soybean_row5_cpd_comparison_result.npy`

## Notes

The Python scripts are intended to reproduce the existing MATLAB logic, using the same input files and hardcoded result vectors where present. If you want the Python versions to parse additional MATLAB data files directly, those scripts can be extended further.
