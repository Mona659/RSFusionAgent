# RSFusionAgent

An executable, traceable agent for remote-sensing spatiotemporal-spectral fusion.

> **V1 scope:** YRE reduced-resolution profile, 151 HS bands, preprocessed HDF5
> inputs and one test patch. Raw-TIFF ingestion and whole-scene stitching are
> deliberately reserved for later versions.

## What V1 does

RSFusionAgent V1 executes a reproducible tool workflow instead of asking a language
model to manipulate image arrays:

```mermaid
flowchart LR
    A["YRE HDF5 pair"] --> B["HDF5 inspector"]
    B --> C["Patch adapter"]
    C --> D["Traceable agent state"]
    D --> E["Isolated PyTorch runtime"]
    E --> F["DC-STSF CUDA inference"]
    F --> G["Metrics tool"]
    F --> H["Artifact tool"]
    G --> I["JSON and Markdown report"]
    H --> J["151-band TIFF, RGB and SAM"]
```

Every step is recorded in `run_manifest.json`. Large arrays are exchanged through
artifact paths rather than placed in agent state.

The architecture is inspired by the task-aware tool orchestration ideas in
[RS-Agent](https://github.com/IntelliSensing/RS-Agent), while this repository
implements an executable workflow for the author's YRE fusion model.

## Verified YRE demo

The V1 workflow was verified end to end on patch index 0 using the epoch-200
checkpoint. The private HDF5 data and checkpoint are not included in this repository.

| Fused RGB preview | Per-pixel SAM heatmap |
|---|---|
| ![YRE fused RGB preview](docs/assets/yre_rgb_preview.png) | ![YRE SAM heatmap](docs/assets/yre_sam_heatmap.png) |

| PSNR (dB) | RMSE | SAM (degree) | ERGAS | SSIM | CC |
|---:|---:|---:|---:|---:|---:|
| 32.0344 | 0.02502 | 2.0984 | 1.8523 | 0.96577 | 0.97729 |

The two independent CUDA runs produced identical Agent outputs. Compared with the
historical TIFF exported by the legacy test script, the maximum absolute pixel
difference was `1.85e-4`, while all six reported metrics agreed at the displayed
precision.

## Frozen YRE-151 data contract

V1 reads the `test` dataset from two legacy HDF5 files. Each patch uses NHWC layout:

```text
[N, H, W, 306]
channels 0:4       -> MS
channels 4:155     -> interpolated auxiliary HS
channels 155:306   -> HS ground truth
```

The model receives:

```text
auxiliary MS : [1, 4,   H,   W] / 10000
auxiliary HS : [1, 151, H/3, W/3] / 10000
target MS    : [1, 4,   H,   W] / 10000
```

See [docs/data_contract.md](docs/data_contract.md) for the complete contract and
known legacy limitations.

## Current capabilities

- [x] Inspect standalone raster metadata
- [x] Validate a YRE-151 auxiliary/target HDF5 pair
- [x] Load and normalize one reduced-resolution test patch
- [x] Run the DC-STSF checkpoint in an isolated PyTorch environment
- [x] Calculate PSNR, RMSE, SAM, ERGAS, SSIM and CC
- [x] Save a 151-band prediction TIFF, RGB preview and SAM heatmap
- [x] Save machine-readable metrics, tool trace and Markdown report
- [ ] Accept three original TIFF inputs
- [ ] Validate pre-registration and geospatial alignment
- [ ] Perform overlapping whole-scene inference and weighted stitching
- [ ] Add optional LLM planning and solution retrieval
- [ ] Add a Gradio interface

## Project structure

```text
RSFusionAgent/
├── docs/
│   ├── assets/
│   │   ├── yre_rgb_preview.png
│   │   ├── yre_sam_heatmap.png
│   │   └── yre_metrics.json
│   └── data_contract.md
├── src/rsfusion_agent/
│   ├── agent/
│   │   ├── state.py
│   │   └── workflow.py
│   ├── models/
│   │   └── dc_stsf.py
│   ├── runtime/
│   │   └── yre151_runner.py
│   ├── tools/
│   │   ├── artifacts.py
│   │   ├── h5_patch.py
│   │   ├── metrics.py
│   │   ├── model_runtime.py
│   │   └── raster_inspector.py
│   └── cli.py
├── examples/
├── tests/
├── outputs/
└── pyproject.toml
```

## Installation

Create the lightweight agent environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The neural network runs in a separate existing Python environment containing a
compatible CUDA-enabled PyTorch installation. This avoids installing a second copy
of PyTorch into the agent environment.

Set the model runtime path for the current terminal session:

```powershell
$env:RSFUSION_MODEL_PYTHON = "C:\path\to\model-env\Scripts\python.exe"
```

Do not commit this machine-specific path. Set it in the terminal session or pass
`--model-python` explicitly. The V1 CLI does not automatically load `.env` files.

## Inspect the HDF5 inputs

```powershell
$auxH5 = "C:\path\to\YRE\DownT1YRE.h5"
$targetH5 = "C:\path\to\YRE\DownT2YRE.h5"

rsfusion inspect-h5 `
  --aux-h5 $auxH5 `
  --target-h5 $targetH5 `
  --patch-index 0 `
  --pretty
```

## Run V1 inference

```powershell
$auxH5 = "C:\path\to\YRE\DownT1YRE.h5"
$targetH5 = "C:\path\to\YRE\DownT2YRE.h5"
$checkpoint = "C:\path\to\checkpoints\model-epochs200.pth"
$modelPython = "C:\path\to\model-env\Scripts\python.exe"

rsfusion infer-h5 `
  --aux-h5 $auxH5 `
  --target-h5 $targetH5 `
  --checkpoint $checkpoint `
  --model-python $modelPython `
  --patch-index 0 `
  --device cuda `
  --output-dir "outputs\yre_patch_0001" `
  --pretty
```

Generated artifacts:

```text
outputs/yre_patch_0001/
├── predicted_hs.tif
├── rgb_preview.png
├── sam_heatmap.png
├── metrics.json
├── report.md
├── run_manifest.json
└── intermediate/
    ├── model_input.npz
    └── model_output.npz
```

V1 intentionally reproduces the legacy TIFF convention: normalized `float32`
values, a synthetic transform and no CRS. This is recorded as a warning in the run
manifest. A future raw-TIFF adapter will preserve target-MS geospatial metadata.

## Tests

```powershell
python -m pytest -q
python -m ruff check --no-cache src tests examples
```

## Security

- Keep API keys and local runtime paths out of Git.
- Load only trusted PyTorch checkpoints. PyTorch checkpoints can contain serialized data.
- Output and intermediate model artifacts are ignored by Git.

## License and attribution

This project is planned to use the Apache-2.0 license. The DC-STSF architecture in
`models/dc_stsf.py` is adapted from the author's research project. Upstream license
and copyright notices must be preserved for any future third-party code reuse.
