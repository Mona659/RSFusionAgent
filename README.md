# RSFusionAgent

An executable agent for remote-sensing image fusion.

> **Status:** early MVP development. The first real tool, raster metadata inspection,
> is available. Fusion planning, algorithms, evaluation and the web demo are being
> implemented incrementally.

## Why this project

Many agent demos stop at selecting a tool. RSFusionAgent is designed to execute a
complete remote-sensing image-fusion workflow: inspect inputs, choose and run a
fusion method, evaluate the output, and produce a reproducible report.

The architecture is inspired by the task-aware solution retrieval and tool
orchestration ideas in [RS-Agent](https://github.com/IntelliSensing/RS-Agent), while
the workflow and executable fusion tools in this repository are implemented for
the image-fusion domain.

## Planned workflow

```text
User request + raster inputs
            |
            v
     Task and input analysis
            |
            v
  Raster validation / alignment
            |
            v
      Fusion tool execution
            |
            v
 Quality evaluation and replanning
            |
            v
    Result artifacts + report
```

## Current capabilities

- [x] Inspect raster metadata through a typed Python tool and CLI
- [x] Return JSON-friendly width, height, bands, dtype, CRS, bounds and resolution
- [x] Generate a tiny demo GeoTIFF for reproducible local testing
- [x] Unit tests for valid and missing raster inputs
- [ ] Raster alignment and resampling
- [ ] Classical fusion methods (Brovey, IHS and PCA)
- [ ] Fusion quality metrics (SSIM, PSNR, SAM and ERGAS)
- [ ] Stateful agent workflow and metric-driven replanning
- [ ] Gradio demo and experiment report

## Project structure

```text
RSFusionAgent/
├── src/rsfusion_agent/
│   ├── cli.py
│   └── tools/
│       └── raster_inspector.py
├── examples/
│   └── create_demo_raster.py
├── tests/
│   └── test_raster_inspector.py
├── data/raw/
├── outputs/
├── .env.example
├── pyproject.toml
└── README.md
```

## Quick start

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Create a small demo raster:

```powershell
python examples/create_demo_raster.py
```

Inspect it through the installed CLI:

```powershell
rsfusion inspect-raster examples/data/demo_multispectral.tif --pretty
```

The command returns structured JSON similar to:

```json
{
  "filename": "demo_multispectral.tif",
  "driver": "GTiff",
  "width": 32,
  "height": 32,
  "band_count": 3,
  "crs": "EPSG:4326",
  "resolution": [0.0001, 0.0001],
  "is_georeferenced": true
}
```

## Tests

```powershell
pytest
```

## Security

Keep API keys in a local `.env` file. The file is ignored by Git; only
`.env.example` should be committed.

## License and attribution

This project is planned to use the Apache-2.0 license. If source code is later
adapted directly from an upstream project, its copyright and license notices must
be preserved.
