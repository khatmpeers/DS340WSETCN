# REPRO_NOTES

## Paper-Confirmed Settings
- Input window size is `10`.
- TCN dilations are `(1, 2, 4)`.
- Kernel size is `4`.
- Hidden channels are `128`.
- Supported model variants are `tcn`, `tcn_r1`, `tcn_r2`, `tcn_r3`, and `setcn`.
- The final best model targeted by the reproduction is `SE-TCN-R3` via `--model setcn`.
- Training years are `2019`, `2020`, and `2021`.
- Test year is `2022`.
- Training loss is MAE.
- Optimizer is Adam.
- Reported evaluation metrics are MAE, RMSE, and MAPE.
- Default retained operational input variables follow the paper-derived Table 1 feature list used in `train.py`.

## Implementation Assumptions
- The plain `tcn` variant uses simpler non-residual temporal layers.
- The residual variants `tcn_r1`, `tcn_r2`, `tcn_r3`, and `setcn` use the deeper residual block implementation in `model.py`.
- `setcn` applies a single SE block after the final residual block.
- Feature scaling is standardization fitted on the training split only and then applied to validation and test features without scaling the target.
- If `--val_split > 0`, validation is taken chronologically from the tail of the 2019-2021 training period and the best validation MAE checkpoint is used for final 2022 testing.

## Remaining Deviations/Open Questions
- The paper description available to this reproduction does not fully specify the internal form of the plain non-residual TCN baseline, so the current `tcn` implementation is a reasonable reproduction assumption rather than a paper-confirmed block design.
- The feature list is enforced as the default paper-faithful set, but column naming must still match the local dataset exactly.
- This repo currently depends on the runtime environment for PyTorch, pandas, and numpy availability; reproducibility also depends on consistent package versions outside the checked-in code.
