# RSFusionAgent 3-minute demo script

This script is for a local interview demonstration. Use the preprocessed YRE HDF5
pair and epoch-200 checkpoint already configured on the demonstration machine. Do not
show an API key, private absolute paths, or raw dataset files on screen.

## 0:00–0:25 — Problem and boundary

> I built a local Agent for YRE spatiotemporal-spectral image fusion. The first version
> supports a verified H5 evaluation route and a raw-TIFF crop-manifest route. The LLM
> orchestrates tools; it never receives image arrays or selects arbitrary local paths.

Open the project README architecture diagram, then launch `rsfusion-ui`.

## 0:25–0:55 — Local safety gate

Show the Streamlit sidebar with the configured H5 pair, checkpoint, model Python and
CUDA device. Click **预检模型环境**.

> Before any paid model request, the program starts the isolated Conda interpreter and
> checks PyTorch, CUDA, GPU visibility and checkpoint readability. If it fails, Qwen is
> never called.

Point out `ready`, the GPU name and `epoch 200`.

## 0:55–1:45 — Agent tool orchestration

Enter the natural-language task:

```text
请先检查数据，再融合第0个patch，并汇报PSNR、SAM、SSIM和输出文件。
```

Click **执行 Agent 融合**.

> Qwen chooses only three strict-schema tools: inspect the H5 pair, run one authorized
> patch, and read the latest result. The DC-STSF model runs in a separate local CUDA
> process; only sanitized metadata and scalar metrics return to the LLM.

## 1:45–2:30 — Results and observability

Show the metric cards, RGB preview, SAM heatmap and tool timeline.

> This verified YRE patch achieves PSNR 32.03, SAM 2.10 and SSIM 0.966. Each run writes
> `preflight.json`, `run_manifest.json`, `metrics.json`, `report.md` and
> `agent_result.json`, including token usage and a cost estimate.

Show that an old run can be loaded under **本地运行历史** without making another API call.

## 2:30–3:00 — Reliability and roadmap

> Native runtime failures are categorized for the Agent. A known Windows fast-fail can
> be retried once and records its attempt count. The TIFF mode uses an explicit crop
> manifest to reproduce legacy source-pixel correspondence, not automatic registration.
> Whole-scene stitching remains a later extension.

Close with the GitHub Actions CI badge and test command:

```powershell
python -m pytest -q
python -m ruff check --no-cache src tests examples
```
