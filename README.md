# TB-SPIN

TB-SPIN integrates single-cell and bulk transcriptomes to predict tuberculosis progression and identify risk cells across the tuberculosis spectrum.

| Directory | Manuscript result |
| --- | --- |
| [Results2](Results2) | TB-SPIN resolves cell states associated with TB infection and TB disease |
| [Results3](Results3) | TB-SPIN predicts progression to TB disease and identifies progression-associated risk cells |
| [Results4](Results4) | TB-SPIN identifies phenotype-associated cells across the TB spectrum and distinguishes asymptomatic from symptomatic TB |
| [Results5](Results5) | TB-SPIN identifies risk cells and associated transcriptional programmes across three TB manifestations |
| [Results6](Results6) | TB-SPIN outperforms established methods in predicting TB progression |

## Running the analyses

The scripts use the study expression matrices, participant splits, fitted models and gene sets as external inputs.

- **Results 2 and 3.** Model modules are in `model`, with training, summary and plotting scripts in `scripts`. The Result 3 training scripts cover the de novo ablation experiment.
- **Result 4.** The scPAS runners and `phenotype` scripts cover cell scoring, enrichment and clinical discrimination. The Python backend requires Eigen headers and can be installed with `python -m pip install ./Results4/backend`. Run `phenotype/clinical_auc.py` on the frozen seed-11 predictions for Supplementary Table 6. The `metabolism` directory contains donor-level pathway analyses. Run `plots/assemble_figure4.py` with the archived panel inputs on Windows with PowerPoint.
- **Result 5.** Run `training/run_fold.py` for folds 0–3 with the original warm-start models and splits. For frozen models and latent tables, run `scoring/score.py` for each seed (20260828–20260830), then `scoring/summarize.py`. The manuscript scores use clipping before exponentiation, and many target cells reach the upper bound.
- **Result 6.** The four comparator runners are in `comparators`. Use `scripts/comparator_programmes.py --help` for score inputs. Set `RESULT6_INPUT` to the directory containing predictions, endpoint observations, expression, programme scores and DEG tables. Run `scripts/benchmark_metrics.py`, then `scripts/plot_manuscript.py`. The `supplement` directory covers representation and threshold sensitivity analyses.

For Figure 5, run these scripts from `Results5` after scoring. The GSEA step uses GSEApy 1.3.1 and MSigDB 2025.1.Hs libraries in `inputs/resources`.

```sh
python analysis/paired_deg_ora.py
for group in DTB TPE TBM; do python analysis/run_gsea.py "$group"; done
python analysis/bidirectional_functions.py
python analysis/render_discrete_NES.py
python analysis/figure/render_upper.py
python analysis/figure/render.py
```

## Verification

Figures 4 and 5 were rendered from archived inputs. For Figure 6, the 24 AUC estimates and both limits of their 95% confidence intervals matched the archived figure source using 2,000 participant-level bootstrap resamples. End-to-end retraining from raw data and pixel-level agreement with the final Word figures remain unverified.
