# YRE-151 V1 data contract

This document freezes the model-facing contract used by RSFusionAgent V1.

## Supported profile

| Field | Value |
|---|---|
| Profile ID | `yre151_reduce_v1` |
| Dataset | YRE reduced-resolution simulation |
| Auxiliary time | T1 / `DownT1YRE.h5` |
| Target time | T2 / `DownT2YRE.h5` |
| HDF5 split | `test` |
| Patch index | configurable, default `0` |
| MS bands | 4 |
| HS bands | 151 |
| Scale | 3 |
| Normalization | divide by `10000.0` |
| Model | `DC_STSF` |

## Legacy HDF5 schema

Both files must expose a numeric NHWC dataset at `test` with shape
`[N, H, W, 306]`. `H` and `W` must be divisible by three.

| Channel range | Meaning | Used by V1 |
|---|---|---|
| `0:4` | Four-band MS | T1 and T2 model inputs |
| `4:155` | Interpolated HS | T1 model input only |
| `155:306` | HS ground truth | T2 metrics; T1 retained for diagnostics |

The T2 `4:155` payload is intentionally ignored because the deployed task predicts
target-time HS from T1 MS/HS and T2 MS.

## Prepared model batch

The HDF5 adapter converts NHWC data to contiguous CHW `float32` arrays and divides
all used channels by `10000.0`.

The isolated PyTorch runtime adds the batch dimension and applies:

```python
auxiliary_hs_lr = torch.nn.functional.interpolate(
    auxiliary_hs_interpolated,
    scale_factor=1 / 3,
    mode="bilinear",
    align_corners=False,
)
```

The model call is:

```python
predicted_hs, reconstructed_auxiliary_hs, latent1, latent2 = model(
    auxiliary_ms,
    auxiliary_hs_lr,
    target_ms,
)
```

Only `predicted_hs` is the primary V1 output.

## Output contract

`predicted_hs.tif` uses shape `[151, H, W]`, dtype `float32`, normalized model
values, no CRS and the legacy synthetic transform `from_origin(0, 0, 1, 1)`.

This output is suitable for regression against the existing test script, not for
GIS positioning. The future TIFF adapter must inherit CRS and transform from the
target-time MS raster and define the restored radiometric scale explicitly.

## Explicitly unsupported in V1

- Raw TIFF ingestion
- Registration or reprojection
- Real/full-resolution `RealT*YRE.h5` profile
- 100-band HZW profile
- Whole-scene overlap and stitching
- Training or checkpoint optimization
- Inference without the legacy HDF5 ground-truth channels

## Future adapter boundary

Future `TiffTripletAdapter` and current `H5PatchAdapter` must both produce the same
framework-neutral fields:

```text
auxiliary_ms
auxiliary_hs_interpolated or auxiliary_hs_lr
target_ms
auxiliary_hs_gt (optional)
target_hs_gt (optional)
spatial_metadata (optional)
```

The Agent workflow and model runtime must not depend directly on HDF5 channel
indices.

