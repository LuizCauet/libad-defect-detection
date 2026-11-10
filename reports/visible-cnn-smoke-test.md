# Visible CNN smoke test

## Purpose

Verify that the paired-image loader, specified CNN, binary loss, AdamW update, validation inference, and checkpoint round trip work together before full training. This is a pipeline diagnostic, not a model-quality result.

## Configuration

- Architecture: `visible_cnn_v1`, 97,425 trainable parameters.
- Input: paired side A/B tensors with shape `2 × 640 × 512`.
- Normalization: fixed statistics from the 446-sample training partition.
- Smoke subset: 16 training samples, balanced as 8 normal and 8 defective.
- Optimizer: AdamW, learning rate `0.001`, weight decay `0.0001`.
- Loss: `BCEWithLogitsLoss`.
- Seed: 17.
- Hardware: NVIDIA GeForce RTX 2060.

## Checks and results

| Check | Result |
| --- | --- |
| Forward output | 16 finite logits |
| Initial loss | 0.691068 |
| Finite gradients | Passed for every trainable tensor |
| Parameter update | Every trainable tensor changed |
| Tiny-subset overfit | Passed in 9 optimizer steps |
| Final tiny-subset loss | 0.048847 |
| Final tiny-subset accuracy | 100% |
| Validation-batch inference | 16 finite logits and finite loss |
| Checkpoint reload | Maximum logit difference `0.0` |
| Peak CUDA allocation | 1,392,908,800 bytes, approximately 1,328 MiB |
| Final-test predictions | None generated |

The validation-batch loss was 1.327709. It is diagnostic only: the model was deliberately trained to memorize sixteen samples, so this value must not be interpreted as validation performance.

## Artifacts

- Model: `src/defect_detection/model.py`.
- Smoke harness: `scripts/smoke_test_visible_cnn.py`.
- Machine-readable result: `artifacts/smoke_visible_cnn_v1/results.json`.
- Step history: `artifacts/smoke_visible_cnn_v1/training_history.csv`.
- Reloaded checkpoint: `artifacts/smoke_visible_cnn_v1/checkpoint.pt`.

## Conclusion

The initial architecture and core learning path are mechanically sound enough to proceed to the full training pipeline. This check does not establish generalization, threshold quality, or improvement over PatchCore.
