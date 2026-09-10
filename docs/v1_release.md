# RSFusionAgent V1.0 release boundary

V1.0 is a local, traceable single-patch fusion Agent for the author's YRE-151 DC-STSF
checkpoint. It is an inference and experiment-demonstration release, not a replacement
for the legacy training pipeline.

## Accepted input routes

1. **Raw TIFF route (primary)** — `T1 MS + T1 HS + T2 MS`, first checked for source
   dimensions, bands, spatial metadata and bounded RGB previews, then cropped through an explicit
   `crop_manifest.json`. It produces a target-MS-georeferenced 151-band prediction TIFF
   and RGB preview. In **real** mode, full-reference metrics are unavailable without `T2 HS`;
   a supplied `T2 HS` is 3× interpolated and clearly labelled only as a pseudo-reference.
   In **legacy Database.py simulation** mode, `T2 HS` is required: its native HS crop is kept as
   reduced-resolution ground truth while the model inputs are blurred and downsampled. This route
   therefore produces valid reduced-resolution metrics and a SAM map.
2. **HDF5 evaluation route (regression)** — one legacy `DownT1YRE.h5` + `DownT2YRE.h5`
   patch with target HS ground truth. The UI checks the `test` dataset, reports the selectable
   Patch count and previews the selected model inputs before inference. It produces PSNR, RMSE,
   SAM, ERGAS, SSIM, CC, RGB and SAM map.

## Verified evidence

- HDF5 patch 0: PSNR `32.0344`, SAM `2.0984`, SSIM `0.9658` with epoch-200 checkpoint.
- Raw TIFF manifest patch 0: YRE legacy crop profile, prediction shape `[151, 180, 180]`,
  CUDA execution with epoch-200 checkpoint, and a georeferenced output TIFF.
- Static quality gate: Ruff and pytest, enforced by GitHub Actions on every push.
- Streamlit retains the latest input-check, preflight, crop and fusion stage result in
  newest-first order and reuses a matching crop manifest without repeating preprocessing.
- After a raw TIFF input check, custom source-crop controls are bounded by compatible input
  dimensions. Source crop size and prepared model-Patch size are independent controls.

## Safety boundary

- Qwen receives only scalar metadata, artifact names and sanitized errors; no image arrays,
  checkpoint contents, API keys or arbitrary local paths are given to the model.
- The raw TIFF manifest is an explicit source-pixel correspondence instruction. It does not
  claim automated image registration or reprojection.
- Native runtime health is checked before an LLM request; the known Windows native
  fast-fail is retried only within a bounded policy.

## Deferred to V1.1 / V2

- TIFF-to-HDF5 training dataset construction
- Automatic registration / reprojection and quantitative alignment evaluation
- Whole-scene sliding-window inference, overlap blending and mosaic export
- Experiment retrieval, model selection and long-term memory
