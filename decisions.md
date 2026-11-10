# Project decision log

This file records the decisions made for the first trainable LIBAD model. Each entry includes the selected approach, Luiz's reasoning, and the alternatives considered.

## Product and experiment scope

| Decision | Selected approach | Luiz's reasoning | Other options considered |
| --- | --- | --- | --- |
| Immediate model scope | Build one visible-light model first and debate additional models later. | I want to control scope and complete one understandable model before expanding the system. | Build visible and X-ray models together; implement multimodal fusion immediately. |
| Initial learning formulation | Train a supervised binary classifier with normal and defective samples. | I expect direct defect supervision to be the simplest useful starting point and to perform well on the known LIBAD defects. | Normal-only PatchCore; one-class feature representation; reconstruction model; eleven-class defect classification. |
| Meaning of the output | Predict whether the physical sample is normal or defective, without predicting the defect category. | I want the first model to detect defects without expanding the task into defect-type classification. | Predict all eleven defect categories; localize defects with segmentation; normal-only anomaly scoring. |
| PatchCore's role | Keep PatchCore as an external reference and compare it fairly later. | I want to establish whether our trainable model offers a useful tradeoff rather than assume that a new architecture is automatically better. | Reimplement PatchCore; reproduce the complete LIBAD paper; omit the reference comparison. |
| X-ray work | Defer X-ray until the visible model is working; later test high-resolution X-ray as a second branch and investigate transfer to low-resolution X-ray. | I think high-resolution X-ray can demonstrate whether X-ray adds useful defect information even if practical acquisition constraints remain unknown. | Exclude high-resolution X-ray as non-inline; begin with low-resolution X-ray; build X-ray fusion in the first milestone. |
| Infrastructure sequence | Finish model and evaluation work before API, Docker, and Kubernetes milestones. | I want infrastructure to demonstrate a working model instead of increasing scope before the ML result is understood. | Build deployment infrastructure alongside the first training pipeline. |

## Data and evaluation protocol

| Decision | Selected approach | Luiz's reasoning | Other options considered |
| --- | --- | --- | --- |
| Split proportions | Use a fixed 60% training, 20% validation, and 20% final-test split. | I accept fewer training samples in exchange for trustworthy validation and test measurements across all defect categories. | 70/15/15; 80/20 without a separate validation set; reuse the normal-only official PatchCore split. |
| Split identity | Freeze `visible_binary_split_v1` with split seed 42: 446 training, 149 validation, and 149 test samples. | I want every model and seed to use the same reproducible sample assignments so changes can be compared directly. | Regenerate a split for each run; use an unrecorded random seed; vary the split together with training initialization. |
| Stratification | Preserve normal/defective and defect-category proportions as closely as sample counts permit. | I want each partition to represent the available categories and avoid accidental class imbalance. | Random splitting without stratification; category-specific models or partitions. |
| Split unit | Keep every physical sample and both of its visible images in one partition. | I want to prevent related A/B images from leaking across training, validation, and test. | Split individual image files independently; treat A and B as separate observations. |
| Final-test access | Keep the 20% test partition sealed until model selection is finished. | I want final results that were not used to guide architecture, preprocessing, threshold, or training changes. | Evaluate the test set after every experiment; turn disappointing test results into additional validation data. |
| Initial-versus-improved comparison | Preserve initial checkpoints and evaluate the initial and selected improved models on the sealed test set at the end. | I want a fair quantified improvement without tuning against final-test behavior. | Test the initial model immediately; compare only validation results; discard the initial checkpoint. |
| Repeated runs | Run every full configuration with training seeds 17, 42, and 73, then report every result plus mean and variation. | I want to know whether improvements occur consistently rather than depend on a favorable initialization. | Report one seed; choose the best of several seeds; use different splits for every run. |
| Operating-threshold rule | Choose the validation threshold with the highest recall while validation FPR is at most 10%. | I prefer a practical balance because a small recall gain is not worth a several-fold increase in falsely flagged good samples. | Fixed threshold 0.5; maximum F1; maximum recall regardless of false alarms; a subjective curve knee. |
| Threshold reporting | Show the complete recall-versus-FPR curve in addition to the selected operating point. | I want the tradeoff to remain visible instead of presenting one threshold as universally correct. | Report only AUROC; report only the chosen confusion matrix; report only maximum F1. |
| Primary quality evidence | Report AUROC, FPR at 90% and 95% recall, the full tradeoff curve, frozen-threshold confusion counts, and per-defect recall. | I want model comparisons to show both ranking quality and the operational cost of catching more defects. | One aggregate accuracy score; precision alone; one overall F1 score. |
| Precision interpretation | State evaluation defect prevalence whenever precision or accuracy is reported. | I want to avoid implying that a defect-heavy benchmark predicts production alert quality when real defects may be rare. | Report precision without prevalence; estimate factory scrap or downtime without production data. |
| Pixel localization | Treat heatmaps as qualitative until suitable ground-truth masks are available. | I want localization claims to be supported by labels rather than inferred from visually plausible heatmaps. | Report pixel metrics as zero or inferred values; claim localization quality from example heatmaps. |

## Input representation and preprocessing

| Decision | Selected approach | Luiz's reasoning | Other options considered |
| --- | --- | --- | --- |
| Visible inputs | Use both visible-light sides A and B for every sample. | I want the model to use evidence from both surfaces because defects may be stronger on one side. | Use side A only; use side B only; make independent predictions and keep only the maximum. |
| Side fusion | Stack A and B as two ordered input channels for early fusion. | The inspected pairs generally share orientation and coordinates, so one CNN can learn features within and across both sides. | Separate shared-weight branches with late fusion; flatten both images into one vector; process sides as unrelated samples. |
| Channel order | Always use side A as channel 0 and side B as channel 1. | I want the learned meaning of each input channel to remain consistent across training and inference. | Randomize side order; average the sides before input. |
| Initial resolution | Resize both sides to height 640 and width 512. | I want a memory-efficient starting point that matches the PatchCore input scale, then I will measure whether higher resolution helps small defects. | Use the original 1,280-by-1,024 resolution immediately; use smaller inputs; use tiles or crops. |
| Pixel scaling | Convert 8-bit values from 0–255 to floating-point values from 0–1. | I want numerically manageable inputs without changing relative image brightness. | Feed raw integer values; normalize every image independently to its minimum and maximum. |
| Statistical normalization | Calculate separate A and B mean and standard deviation values from training images only, then reuse them for every partition. | I want inputs expressed relative to normal training statistics while preserving unusually bright or dark images and avoiding leakage. | Per-image standardization; shared A/B statistics; statistics calculated from the complete dataset; scaling to 0–1 only. |
| Initial augmentation | Use no augmentation in the initial model. | I want to establish whether the basic network learns before testing whether synchronized flips provide measurable value. | Permanent flipped copies; random synchronized horizontal and vertical flips; rotations, crops, or brightness changes. |
| Future flip experiment | Consider synchronized on-the-fly horizontal and vertical flips as a controlled ablation. | I want to measure whether positional invariance improves all three seeds rather than assume that transformed copies add useful information. | Add flips by default; augment validation and test data; create transformed files before splitting. |

## Network architecture

| Decision | Selected approach | Luiz's reasoning | Other options considered |
| --- | --- | --- | --- |
| Backbone | Train a small custom CNN from random initialization. | I want to understand and own a complete trainable architecture instead of only running a downloaded feature extractor. | Frozen pretrained backbone; fine-tuned DINO or ResNet; PatchCore features. |
| Number of stages | Use four convolutional stages. | I want a simple network with enough depth to build features progressively without making the first implementation difficult to understand. | One to three stages; a deeper residual network; transformer blocks. |
| Channel progression | Increase channels as `2 → 16 → 32 → 64 → 128`. | I agree with learning more feature types as spatial dimensions become smaller and memory becomes cheaper. | Constant channel count; a wider starting layer; a narrower or deeper progression. |
| Convolution settings | Use 3-by-3 kernels, stride 1, padding 1, and a bias. | I want a standard local feature extractor that preserves spatial size before explicit pooling. | Larger kernels; stride-2 convolutions; valid convolution without padding; dilated convolutions. |
| Activation | Apply ReLU after every convolution. | I want a common, computationally simple nonlinear activation for the initial network. | GELU; SiLU; Leaky ReLU; no activation. |
| Spatial downsampling | Apply 2-by-2 max pooling after each ReLU, halving height and width at every stage. | I want separate, understandable operations for increasing channels and reducing spatial dimensions. | Average pooling; stride-2 convolution; preserve full resolution; adaptive pooling after fewer stages. |
| Internal normalization | Use no batch, group, or layer normalization initially. | I want the first blocks to stay simple and to add internal normalization only if training evidence supports it. | Batch normalization; group normalization for small batches; layer normalization. |
| Final feature aggregation | Use adaptive global max pooling to reduce `128 × 40 × 32` to 128 values. | I care about detecting a strong defect feature even when it occupies only a small part of the image. | Global average pooling; combined average and maximum pooling; flatten the full feature map. |
| Classifier head | Use one direct linear layer from 128 pooled values to one logit. | I want the simplest classifier and will add capacity only if the model underfits. | Hidden dense layer with ReLU; dropout; a large flattened dense classifier; two-class softmax. |
| Output activation | Train on the raw logit and apply sigmoid only when a 0–1 score is needed. | I want a stable binary-loss implementation and a score suitable for threshold analysis. | Two-output softmax; ReLU output; sigmoid as a separate training layer followed by ordinary binary cross-entropy. |
| Initial parameter count | Keep the specified model at 97,425 trainable parameters. | I want a small model whose behavior is easier to diagnose on a dataset of only 744 samples. | Increase width or add dense layers before observing underfitting. |

## Training decisions

| Decision | Selected approach | Luiz's reasoning | Other options considered |
| --- | --- | --- | --- |
| Loss | Use unweighted `BCEWithLogitsLoss`. | The complete dataset is almost balanced, so I do not want to introduce class weighting without evidence that it is needed. | Weighted binary cross-entropy; focal loss; multiclass cross-entropy; one-class losses. |
| Optimizer | Use AdamW for the initial model. | I want a reliable first optimization run so that optimizer tuning does not obscure whether the architecture can learn. | Plain SGD; SGD with momentum; Adam; RMSprop; Adagrad. |
| AdamW settings | Start with learning rate `0.001` and weight decay `0.0001`. | I want common starting values that are likely to train stably while treating them as tunable settings rather than proven optima. | Other learning rates; no weight decay; stronger regularization. |
| Learning-rate schedule | Use a constant learning rate initially. | I want to observe the basic optimizer behavior before adding another mechanism. | Step decay; cosine decay; reduce-on-plateau; warmup. |
| Batch size | Train with batch size 16, with 8 as an explicit out-of-memory fallback. | The measured batch-16 training step used about 2,056 MiB of global VRAM on a 6,144 MiB RTX 2060, so it fits while providing stable gradients. | Batch sizes 4 or 8; use the maximum possible batch; gradient accumulation. |
| Training limit | Allow at most 100 epochs. | I want enough time for convergence without allowing an unbounded training run. | Fixed shorter training; continue until manual interruption; larger epoch limits. |
| Early stopping | Stop after 10 completed epochs without validation-loss improvement. | I want to limit overfitting and unnecessary computation when validation performance has stopped improving. | No early stopping; patience of another length; stop on AUROC instead of loss. |
| Checkpoint selection | Restore the checkpoint with the lowest validation loss. | I want a predefined selection rule instead of using whichever weights exist at the final epoch. | Highest validation AUROC; best F1; latest checkpoint; lowest training loss. |
| Initialization | Use PyTorch's default initialization and control it with recorded random seeds. | I want a standard reproducible starting point before testing initialization as a separate factor. | Custom Kaiming initialization; pretrained weights; identical fixed weights for every experiment. |

## Workflow and tooling

| Decision | Selected approach | Luiz's reasoning | Other options considered |
| --- | --- | --- | --- |
| Smoke test | Validate loading, shapes, gradients, small-subset overfitting, and checkpoint reload before full training. | I want to catch pipeline bugs before spending time on experiments or interpreting model quality. | Start all three full runs immediately; rely only on unit tests; inspect failures after training. |
| Initial logging | Use ordinary JSON, CSV, and checkpoint files during the smoke test. | I want basic training bugs to remain visible without adding experiment-tracking setup at the same time. | Configure MLflow before any model execution; keep console output only. |
| Full experiment tracking | Use local MLflow with a SQLite tracking database and local artifact directory for every full three-seed run after the smoke test. | I want reproducible comparisons that connect settings, metrics, curves, and checkpoints for each experiment. | MLflow's deprecated filesystem tracking backend; manually named result folders only; cloud MLflow server; another platform such as Weights & Biases. |
| Change discipline | Change one meaningful factor at a time and compare the same three seeds. | I want to know which change caused an improvement and whether the effect is consistent. | Change architecture, optimizer, and augmentation together; keep only successful runs. |
| Result interpretation | Treat lower FPR obtained through materially lower recall as a tradeoff, not an unconditional improvement. | I want resume claims to compare models at meaningful operating conditions rather than hide missed defects. | Compare each model at its own best accuracy or F1 without matching recall. |

## Planned first implementation sequence

1. Generate and validate the fixed stratified split manifest.
2. Implement paired visible-image loading and training-only normalization statistics.
3. Implement the four-stage CNN and binary loss.
4. Implement training, validation, early stopping, checkpointing, and threshold analysis.
5. Complete the smoke test.
6. Add local MLflow tracking.
7. Run three full training seeds while keeping the final test sealed.
8. Review validation curves, per-category behavior, false positives, and false negatives.
9. Select and test one improvement hypothesis at a time.
10. Evaluate preserved initial and selected improved models on the sealed final test only after model selection ends.
