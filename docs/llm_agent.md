# LLM control plane

RSFusionAgent V0.2.1 adds an optional natural-language control plane built on an
OpenAI-compatible Responses API function-calling loop. The language model chooses
from a small allowlist; NumPy arrays and raster pixels stay inside local deterministic
tools.

## Allowlisted tools

| Tool | Purpose | Safety rule |
|---|---|---|
| `inspect_yre151_h5` | Validate the configured HDF5 pair | Must succeed before inference |
| `run_yre151_fusion` | Run one configured patch | Cannot change local paths or patch index |
| `get_latest_fusion_result` | Read the latest metrics and artifact names | Requires a completed run |

Every schema uses strict JSON arguments and rejects additional properties. The model
cannot choose arbitrary filesystem paths: paths are supplied by the CLI and held in
local context. Only file names, shapes, scalar statistics, metrics and warnings are
returned to the API. Configured paths are redacted from tool errors. Image arrays,
HDF5 content and checkpoints are never uploaded.

## Loop

```mermaid
sequenceDiagram
    participant U as User
    participant L as LLM control plane
    participant T as Local toolbox
    participant M as DC-STSF runtime
    U->>L: Natural-language request
    L->>T: inspect_yre151_h5
    T-->>L: Sanitized metadata
    L->>T: run_yre151_fusion
    T->>M: Local CUDA subprocess
    M-->>T: Prediction artifact path
    T-->>L: Metrics and artifact names
    L-->>U: Final answer and local result
```

The loop is bounded by `--max-turns` and records tool call IDs, validated arguments,
status, elapsed time and sanitized output. Tool failures are returned to the model so
it can explain or safely recover.

## Provider configuration and observability

Use `--provider qwen` with the generic environment variables below for Model Studio's
OpenAI-compatible endpoint. The model identifier and base URL may instead be supplied
with the CLI flags of the same names.

```powershell
$env:RSFUSION_LLM_PROVIDER = "qwen"
$env:RSFUSION_LLM_API_KEY = "your-api-key"
$env:RSFUSION_LLM_BASE_URL = "https://<workspace-id>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
$env:RSFUSION_LLM_MODEL = "qwen3.7-flash"
```

The adapter also accepts `openai`, `deepseek`, and `custom` providers. For backwards
compatibility, it falls back to `DASHSCOPE_API_KEY`, `DEEPSEEK_API_KEY`, or
`OPENAI_API_KEY` as appropriate. API keys are environment-only: neither tool calls nor
run artifacts expose them.

Every completed run writes `<output-dir>/agent_result.json`. It records the provider,
model, request, tool trace, model-response trace, normalized token totals, and a
best-effort cost estimate when a local pricing rule exists. Cost estimates are not an
invoice: confirm actual charges in the provider console, particularly when cache or
long-context pricing applies.

## Security notes

- Set `RSFUSION_LLM_API_KEY` (or a provider-specific fallback) in the environment;
  never put it in CLI arguments or Git.
- Only load trusted PyTorch checkpoints.
- Treat the natural-language layer as a control plane, not a numerical compute layer.
- Review the configured paths and patch index before starting a paid API request.
