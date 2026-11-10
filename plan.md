# Project plan: battery electrode defect detection

## Purpose

Build an inspection system that finds defects in battery electrode images. Use the project to show employers that Luiz can inspect data, train models, compare results, diagnose errors, and deliver software that another engineer can run.

The main development work is building or adapting our own trainable model. We will compare that model with an existing method and test improvements. Beating the researchers is not a requirement for a successful project.

LIBAD contains lithium-ion electrode images. Elevated Materials uses a different lithium-metal manufacturing process. The project demonstrates transferable inspection skills; it does not establish performance on their production line.

## How we will work

- Luiz runs training and evaluation commands and examines the output.
- The assistant explains the commands, results, and relevant code.
- We discuss important choices before changing the model, data split, preprocessing, or evaluation rules.
- We record each experiment's question, change, configuration, result, and conclusion.
- We distinguish measured results from goals and possible explanations.

## Terms used in this plan

| Term | Meaning |
| --- | --- |
| Baseline | An existing method that we run as a reference for later comparisons |
| Feature extractor or backbone | A network that converts an image into numerical descriptions of visual patterns |
| Frozen backbone | A backbone whose weights do not change during fitting |
| Fine-tuning | Updating some or all pretrained weights with our training data |
| Memory bank | A collection of normal feature vectors used for comparison |
| Coreset | A representative subset of the extracted feature vectors |
| Recall | The fraction of actual defects that the system flags |
| Precision | The fraction of flagged samples that actually contain defects |
| False-positive rate (FPR) | The fraction of good samples that the system incorrectly flags |
| FPR95 | FPR at the first evaluated score threshold that reaches at least 95% defect recall |
| Image AUROC | Area under the receiver operating characteristic curve; measures how well scores rank defects above normal samples across thresholds |
| Latency | Time required to process an input and return a result |
| Throughput | Number of samples processed per second |
| p95 latency | The processing time at or below which 95% of measured requests finish |
| Ablation | An experiment that removes or changes a component to measure its contribution |
| Inference | Using a fitted model to produce predictions |

## Current position

We have completed a first development baseline using the existing LIBAD code. The run used public DINOv2-small features because the default DINOv3 weights require separate authenticated access.

The run used 27 normal training samples and 49 evaluation samples from the wrinkling category. Image AUROC was approximately 0.696. Reaching at least 95% defect recall required flagging all 11 normal evaluation samples.

This result applies to one category, one split, and our selected configuration. It is not a reproduction of the paper's headline result. See `reports/baseline-results.md` for the full settings, measurements, and limitations.

Next, inspect the images and heatmaps to understand the overlapping normal and defect scores. Finish the evaluation design before using those findings to select model changes.

## Tools and their purpose

| Task | Tools | What the work should demonstrate |
| --- | --- | --- |
| Build and train models | Python, PyTorch, torchvision | Understand the training loop, pretrained features, fine-tuning, and saved model states |
| Load images and calculate metrics | NumPy, pandas, scikit-learn, OpenCV | Check data, implement preprocessing, and evaluate each defect type |
| Track experiments | MLflow, initially stored locally | Save settings, metrics, and generated files for each run |
| Reproduce runs | Git, pyproject.toml, version locks, YAML configuration files | Install a known environment and rerun an experiment with documented commands |
| Check software | pytest, Ruff, GitHub Actions | Test important behavior and run automated code checks |
| Provide predictions through a service | FastAPI and Docker | Define inputs and outputs, handle errors, and package dependencies |
| Manage running service copies | Kubernetes, later | Deploy, recover, update, and roll back the service |
| Optional inference optimization | ONNX Runtime; TensorRT if NVIDIA deployment needs justify it | Improve measured performance and check prediction consistency |
| Optional remote compute | One Amazon Web Services (AWS) graphics processing unit (GPU) workflow | Run reproducibly, record costs, and shut down unused resources |

These tools support specific tasks. We do not need to use every tool immediately. The current environment uses a Python virtual environment and a pip package-version file; we can adopt uv and project metadata when we organize the shared pipeline.

## Milestone 1: Check the data and run a reference method

### Work

1. Confirm dataset access, license terms, hardware, image formats, and available labels.
2. Count normal and defective samples in each category and split.
3. Check whether images share a sheet, roll, or acquisition group. Related images across splits can make evaluation easier than deployment.
4. Record a file inventory and sample identifiers. Keep dataset files and credentials out of Git.
5. Run an existing PatchCore implementation on visible images for one category and one official split.
6. Record the source revision, configuration, software versions, and hardware.
7. Save image scores, heatmaps, the memory bank, metrics, and errors.
8. Explain the output and inspect normal images and defects alongside their heatmaps.

Running the baseline should be a small reference experiment. Rewriting PatchCore or implementing the paper's DA-Core method is optional. We do not need to reproduce every table in the paper.

Compare our result with a published result only when the input modalities, splits, preprocessing, and evaluation rules match. Investigate substantial unexplained differences before making claims.

### Completion evidence

A documented command produces a baseline report. The report states which data the model used and what the measurements mean.

Record sample counts, AUROC, FPR95, runtime, peak memory, and any observed variation between repeat runs. Sample counts describe the work's scope; they do not establish model quality.

## Milestone 2: Build our trainable model and compare it fairly

### Work

1. Understand the reference's feature extraction, memory bank, and scoring well enough to explain and debug their use.
2. Create a small shared interface for fitting models and producing predictions. Reuse data loading and evaluation code where appropriate.
3. Choose a different trainable approach after inspecting the available labels:
   - With normal-only training, consider a teacher-student method or a reconstruction model. A teacher-student method learns to match a reference network; a reconstruction model learns to reproduce its input.
   - With suitable defect labels, consider supervised fine-tuning. The available labels determine whether classification or pixel-level segmentation is feasible.
4. Understand and implement the selected model's data processing, loss function, training loop, saved model states, and inference code.
5. Save the first working version as the starting point for our improvement experiments.
6. Track experiments in MLflow. Record which labels, pretrained models, and tuning resources each approach uses.

A pretrained backbone is acceptable. Ownership means understanding, modifying, and debugging the system; it does not require training every component from scratch.

### Completion evidence

Both approaches run through a documented evaluation process. Luiz can explain how each produces a score and how the trainable model updates its weights.

Compare models on the same evaluation samples. Measure speed on the same hardware with the same timing boundaries. If one model uses defect labels and another does not, describe the result as a comparison of complete learning approaches rather than attributing every difference to architecture.

Record per-type AUROC and FPR95, recall and FPR at a fixed validation-selected threshold, processing time, memory, and fitting cost. Check promising conclusions across additional runs or splits.

## Milestone 3: Diagnose errors and test an improvement

### Work

1. Group observed errors by a supported explanation. Candidates include tiny defects, normal texture variation, lighting differences, missing context, and defects that are weak in the chosen imaging mode.
2. Write a hypothesis before changing the system.
3. Change one meaningful factor when testing that hypothesis.
4. Select changes using development data. Check promising results across multiple runs or splits.
5. Run ablations to determine which components explain a gain. Keep records of changes that did not help.

Possible experiments include higher image resolution, overlapping crops, different feature layers, broader coverage of normal variation, and fine-tuning.

### Completion evidence

Show a supported improvement over our initial model, or explain why the tested changes did not help. Separately report how the final model compares with PatchCore.

Measure FPR at comparable recall, changes for individual defect types, runtime, and memory. Use the same samples for paired comparisons. Account for related samples when estimating uncertainty.

For example, reducing FPR from 40% to 25% is a 15-percentage-point reduction and a 37.5% relative reduction. The comparison must state whether the starting point was our initial model or the external baseline.

A lower FPR obtained by accepting much lower recall is a tradeoff, not an improvement at equal recall. We do not promise a numerical gain before the experiments.

## Milestone 4: Improve speed and memory use

### Work

1. Define a simulated processing-time and memory budget. Actual factory requirements are unknown.
2. Measure image decoding, preprocessing, feature extraction, feature search, and output processing separately.
3. Improve the measured bottleneck. Options include bank size, search implementation, batch size, lower-precision computation, and model export.
4. Check whether scores, threshold decisions, or per-type metrics change after optimization.

Exporting the feature extractor does not automatically export PatchCore's memory-bank search. Include every required component in the performance comparison.

### Completion evidence

Produce a quality, speed, and memory comparison on named hardware. Document warmup, GPU synchronization, batch size, and the number of simultaneous requests.

Measure median and p95 full-pipeline latency, throughput, and peak memory. Distinguish these from timing only the model's forward pass.

Do not convert images per second into production-line speed without knowing the physical image coverage and line specifications.

## Milestone 5: Package a service another engineer can run

### Work

1. Build a FastAPI application programming interface (API) for predictions.
2. Use Docker to package the service and its dependencies.
3. Document accepted inputs, returned outputs, and model/configuration versions.
4. Validate inputs and handle errors. Return a score, threshold decision, and optional heatmap.
5. Add health and readiness checks.
6. Test important behavior: split separation, metric calculations, preprocessing, model loading, and API responses.
7. Run lightweight automated checks in GitHub Actions using synthetic test data. Run expensive GPU evaluation separately.
8. Send evaluation inputs through the service and measure its performance under stated load.
9. Demonstrate monitoring with controlled brightness or blur changes. Such changes are warning signals; labels are needed to establish an accuracy drop.

### Completion evidence

Another engineer can install or start the service, send an example request, reproduce a report, and find the limitations.

Record p95 request latency, throughput, errors, and behavior for invalid inputs. Verify installation in a clean environment. Describe this as a deployment demonstration, not a factory deployment.

## Milestone 6: Demonstrate Kubernetes deployment and recovery

Kubernetes manages running containers. After the Docker service works, use kind or minikube to create a local cluster.

### Work

1. Define a Deployment to manage running copies of the API and a Service to route requests.
2. Set resource requests and limits. Configure checks for startup, readiness to receive requests, and whether the process needs restarting.
3. Verify predictions against the standalone service with the same model files.
4. Terminate one running copy, called a replica. Measure replacement time and request failures during recovery.
5. Compare one replica with multiple replicas under fixed load.
6. Deploy a new model version, then demonstrate rollback to the previous version. Check version identifiers and request failures throughout.

### Completion evidence

Save deployment configuration files, startup and shutdown commands, and a measured recovery/load report. Record p95 latency, throughput, error rate, and total resource use.

Replicas on one machine share finite hardware resources. They may not improve throughput. A local single-machine demonstration does not establish recovery from physical host failure.

Automatic scaling and a managed cloud cluster remain optional.

## Milestone 7: Present the evidence to employers

### Deliverables

- A README with the problem, measured findings, example outputs, and setup commands.
- A technical report with the data rules, model comparisons, ablations, error analysis, speed/memory tradeoffs, and limitations.
- A table linking each headline result to its configuration, saved output, and evaluation command.
- A short demonstration video and system diagram.
- A model card that describes intended use, evaluation, and limitations.
- Dataset attribution and license notes.
- Small reviewable commits, tracked tasks, and explanations of important decisions.
- An interview walkthrough in which Luiz explains and modifies critical code.

### Choose the resume story from the results

If our model offers a useful tradeoff against PatchCore, lead with that comparison. State any disadvantages.

Otherwise, lead with verified improvements over our own initial model, code ownership, and software delivery. Name the comparator explicitly. Keep the external baseline results in the report even if PatchCore wins.

A supported recommendation to use the existing method is a valid engineering conclusion. The project demonstrates skills; it cannot guarantee employment or immediate readiness for every role.

## Rules for trustworthy evaluation

- Inspect the official split definitions before creating validation data. Disclose any change to the benchmark protocol.
- Define a development set and reserve independent final evaluation data before further model selection. The category already inspected is development evidence.
- Do not assume a different split seed provides untouched data; official splits can share samples.
- The inspected validation protocol contains only normal samples. It cannot directly select a threshold for a target defect recall without additional labeled defect development data. We must discuss and document the approach before tuning.
- Distinguish FPR95 calculated from an evaluation curve from performance at a threshold chosen on validation data. Freeze the deployment threshold before final testing and report actual test recall and FPR.
- Report sample counts, per-category results, and how aggregate metrics are calculated.
- State whether thresholds are global or category-specific. Separate category thresholds do not demonstrate one shared operating threshold.
- Quantify localization accuracy only when suitable defect masks exist. Otherwise treat heatmaps as qualitative examples.
- Consider how common defects are when interpreting precision and review workload. FPR alone does not establish scrap, downtime, or cost savings.
- Account for small sample counts and related images when discussing uncertainty.

## Sequence and approximate effort

Complete the data work, model comparison, and improvement experiments before expanding infrastructure. Kubernetes follows the working Docker API. Cloud deployment and multimodal experiments remain optional.

| Work | Initial estimate in work blocks |
| --- | ---: |
| Data audit and reference baseline | 1–2 |
| Our trainable model and shared evaluation code | 3–5 |
| Improvement experiments and ablations | 2–4 |
| Performance optimization and service | 2–3 |
| Local Kubernetes demonstration | 1–2 |
| Documentation and presentation | 1–2 |

These ranges are estimates, not deadlines. Revise them as we learn from the data, hardware, and implementation.

An optional extension compares visible-light images with low-resolution X-ray and their combination. High-resolution X-ray supports annotation and analysis rather than the benchmark's inline-compatible inputs.

## Resume templates

Replace placeholders only with verified measurements. Keep the reference result and measurement conditions available in the report.

- Built a reproducible PyTorch inspection pipeline comparing [N] approaches across [K] defect types and [M] samples.
- Reduced FPR from [A]% to [B]% at [R]% recall relative to [our initial model / named reference] through [specific change], evaluated under [protocol].
- Achieved [X]× faster full-pipeline inference than [reference] on [hardware], with [explicit detection tradeoff].
- Reduced p95 latency from [A] to [B] ms while maintaining [specified quality tolerance].
- Delivered a Dockerized FastAPI service with automated checks and [measured load-test result].
- Deployed the service on Kubernetes and measured [recovery time] and [request error rate] under [specified load].

## References and hiring context

The job postings below informed the original roadmap. They are examples of relevant responsibilities, not a comprehensive hiring survey. Several are senior roles. Their requirements do not establish entry-level eligibility, and posting availability can change.

- FieldAI: model training, deployment, PyTorch/TensorFlow, and OpenCV: https://jobs.lever.co/field-ai/b600a2ab-cff1-4b8f-9940-4bd851c05e37
- Torc: production-quality Python/PyTorch and vision deployment: https://job-boards.greenhouse.io/torcrobotics/jobs/8765533002
- Woven by Toyota: training, data quality, model tradeoffs, and deployment constraints: https://jobs.lever.co/woven-by-toyota/d59b0f20-4b3b-49bf-b9a6-600ee7050db0
- Zoox: checking predictions after optimization and measuring inference latency: https://jobs.lever.co/zoox/f2da6b2e-791a-49b1-87e6-d0313b105507

Project sources:

- LIBAD code: https://github.com/evenrose/LIBAD
- LIBAD paper: https://arxiv.org/abs/2608.07958
- PatchCore paper: https://arxiv.org/abs/2106.08265
