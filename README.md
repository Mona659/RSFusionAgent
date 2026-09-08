# RSFusionAgent

An executable, traceable agent for remote-sensing spatiotemporal-spectral fusion.

> **V1 scope:** YRE reduced-resolution profile, 151 HS bands, preprocessed HDF5
> inputs and one test patch. Raw-TIFF ingestion and whole-scene stitching are
> deliberately reserved for later versions.

## What V1 does

RSFusionAgent V1 executes a reproducible tool workflow instead of asking a language
model to manipulate image arrays:

```text
DownT1YRE.h5 + DownT2YRE.h5 + checkpoint
                    |
                    v
           inspect_h5_dataset
                    |
                    v
          prepare_yre151_patch
                    |
                    v
           run_dc_stsf_patch
                    |
                    v
           validate_prediction
                    |
                    v
       evaluate + save artifacts + report
```

Every step is recorded in `run_manifest.json`. Large arrays are exchanged through
artifact paths rather than placed in agent state.

The architecture is inspired by the task-aware tool orchestration ideas in
[RS-Agent](https://github.com/IntelliSensing/RS-Agent), while this repository
implements an executable workflow for the author's YRE fusion model.

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

For the current workstation:

```powershell
$env:RSFUSION_MODEL_PYTHON = "C:\Users\think\.conda\envs\zmj310\python.exe"
```

Do not commit this machine-specific path. Set it in the terminal session or pass
`--model-python` explicitly. The V1 CLI does not automatically load `.env` files.

## Inspect the HDF5 inputs

```powershell
rsfusion inspect-h5 `
  --aux-h5 "D:\file_zmj\dataset\cx\YRE\DownT1YRE.h5" `
  --target-h5 "D:\file_zmj\dataset\cx\YRE\DownT2YRE.h5" `
  --patch-index 0 `
  --pretty
```

## Run V1 inference

```powershell
rsfusion infer-h5 `
  --aux-h5 "D:\file_zmj\dataset\cx\YRE\DownT1YRE.h5" `
  --target-h5 "D:\file_zmj\dataset\cx\YRE\DownT2YRE.h5" `
  --checkpoint "D:\file_zmj\projects\001NET1\result_YRE\0813-2008\backup_models\model-epochs200.pth" `
  --model-python "C:\Users\think\.conda\envs\zmj310\python.exe" `
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
pytest
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
