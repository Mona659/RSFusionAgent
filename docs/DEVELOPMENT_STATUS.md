# Development Status

本文件记录当前仓库做到哪里。判定优先级为：当前代码与测试 > 当前配置 > 已跟踪文档 > Git 历史 > 未跟踪历史总结。这里的“完成”表示代码和相应自动化证据已存在，不代表所有真实数据、GPU、浏览器和外部供应商组合都已验证。

审查基线：`main` 分支，提交 `dc8182b`（2026-09-17）。

## Completed

### Core data and inference workflows

- H5 YRE151 Patch 数据检查、通道拆分、归一化、单 Patch 推理、指标、预览、清单和报告。
- 原始 GeoTIFF 三元组/四元组检查、RGB 预览、YRE 预设或自定义窗口裁剪、可追溯 `crop_manifest.json`。
- TIFF 真实实验和模拟实验的单 Patch 推理，保留目标 MS 地理参考，并按参考可用性计算指标。
- `external_registration` 输入模式：允许已由外部流程配准的数据在元数据仍有差异时继续，但明确保留警告且不声称自动配准。
- DC_STSF 运行时契约（4 MS、151 HS、3 倍尺度）及严格 Checkpoint 加载。
- Agent/UI 环境与 PyTorch/CUDA 模型环境分离；预检、超时和已知 Windows 原生崩溃的有限重试已实现。

### Agent and control plane

- 自定义有界、顺序 Function Calling 循环，关闭并行工具调用。
- OpenAI/Qwen/DeepSeek/Custom provider 配置和 OpenAI-compatible Responses API 客户端。
- Qwen 兼容 Base URL 到 Responses 端点的归一化处理。
- 查询、H5、TIFF 三类工具箱及代码级工具 allowlist。
- 本地路径注入、工具前置依赖、显式融合授权和 TIFF 清单授权。
- 规则优先、LLM 回退的意图路由，覆盖知识、状态、输入、裁剪和融合意图。
- 轻量 `TaskState` 持久化、基于已完成工具的执行计划和常见失败恢复建议。
- 可选 LangChain `StructuredTool` 适配层；核心执行不依赖 LangGraph。

### Knowledge, diagnostics and observability

- `knowledge/` 的离线 Markdown/JSON 加载、切块和词法 top-k 检索。
- 独立错误知识检索和历史 `agent_result.json` 实验摘要检索。
- 版本化 RAG 评测用例、跨 Windows/POSIX 来源路径归一化和来源命中率报告。
- 配置、数据、CUDA、原生库、超时、LLM 和未知错误分类。
- 脱敏 `execution_trace.json` 与 `agent_execution_report.md`，包含工具状态、耗时、参数字段、Token/成本摘要、状态、恢复建议和制品文件名。
- `scripts/recover_agent_result.py` 可从既有 H5 `run_manifest.json` 重建 Agent 结果；它是显式离线恢复工具，不是自动错误回退。

### Interfaces and engineering baseline

- CLI 子命令：栅格/H5/TIFF 检查、裁剪、预检、H5/TIFF 推理、三类 Agent 入口和 RAG 评测。
- Streamlit 本地 UI：输入配置、自然语言请求、快捷操作、结果展示、当前会话对话、历史本地结果读取和非秘密偏好保存。
- 旧结果修改时间检查，防止本轮失败时展示上轮 `agent_result.json`。
- Ruff、pytest 和 GitHub Actions Python 3.10 CI 配置。
- 20 个测试模块覆盖 Agent、工具、状态、计划、RAG、可观测性、H5/TIFF、指标、运行时桥接和 UI 辅助逻辑。

## In Progress

以下内容在代码中已存在，但因近期持续修改或验证边界不足，不应描述为完全稳定：

1. **Streamlit 交互与布局**：最近两个提交集中修改配置持久化、对话区和页面布局。辅助逻辑有 pytest 覆盖，但没有真实浏览器端到端/响应式布局测试。当前用户已要求暂时忽略布局问题，因此本轮不修改 UI。
2. **LLM 供应商配置兼容**：Qwen Responses 端点和模型配置刚在 `dc8182b` 调整。客户端逻辑和测试存在，但外部模型可用性、免费额度、实际模型 ID 和服务端错误取决于用户账户与供应商，仓库无法保证。
3. **版本/发布文档统一**：作者已确认当前 Python 包正式版本为 `1.1.1`、项目正式开发版本为 V2.1，但 README 和历史文档尚未统一表达这两个不同层级的版本标识。
4. **当前文档同步**：本次已新增五份持久化上下文文档，但 README、`docs/data_contract.md`、`docs/llm_agent.md` 和部分源码 docstring 仍保留“未来 TIFF 适配器”或旧工具列表等历史表述。根据本轮要求，未修改这些现有文件。

## Pending

以下事项来自当前已跟踪 README/`docs/v1_release.md` 的明确未完成项或仓库缺口；没有据此推断排期：

- TIFF 三元组到旧版 HDF5 训练数据集的构建。
- 自动配准、重投影和像素级/定量对齐评估。
- 整景滑窗推理、重叠加权和 Mosaic 导出。
- 通用模型选择/多模型适配。当前只有内置 YRE DC_STSF 运行时契约。
- 明确应用代码与未开源 DC_STSF 模型代码的许可边界；当前还没有可直接执行的许可证方案。
- 对当前 README 与旧技术文档做一次不改变业务逻辑的内容同步。

注意：`docs/v1_release.md` 还把实验检索和长期记忆列为早期 deferred 项。作者已确认当前项目中的“长期记忆”仅指基于历史 `agent_result.json` 的本地实验摘要检索；该范围已经实现，不包含额外的跨项目向量记忆或服务化记忆需求。

## Known Issues

### Code and behavior

- `TaskStage.INFERENCE_AUTHORIZED` 已定义但当前无状态转换写入；`PlanStepStatus.NOT_REQUIRED` 已定义但计划构建器不产生该状态。它们可能是预留值，删除或接入前需要兼容性决策。
- 单一 `TaskStage` 不能完整表达“输入检查”和“运行时预检”这类可独立完成的维度；当前真实依赖判断依赖 `completed_tools`，只展示 `stage` 时可能造成误读。
- 工具 Schema 在原生工具箱与可选 LangChain 包装之间需要人工同步，新增/改名工具存在遗漏风险。
- Streamlit 对话记录只保存在 Session State，服务或浏览器会话重启后不会形成持久对话存档。
- Streamlit 的浏览器布局和滚动行为缺少自动化端到端验证。
- 外部配准模式不测量残余像素偏差；它仅接受用户对配准关系的声明，并把特定元数据差异降级为警告。
- 本地成本估算仅覆盖代码中硬编码的少数模型/价格，不能作为供应商账单依据。

### Documentation and repository

- README 的 capability checklist 仍把 solution retrieval 和 experiment memory 标为未完成，但代码中已有 `error_retriever.py` 和 `experiment_memory.py`。
- README、`docs/data_contract.md` 及 `tools/tiff_triplet.py` 的部分文字仍称 TIFF adapter 为 future/planned，与当前 TIFF 工作流不一致。
- `docs/llm_agent.md` 的早期工具表未覆盖当前 RAG、状态、预检和 TIFF 工具。
- README 链接的 `docs/v1_development_summary.md` 及同类 V2 计划/总结当前为未跟踪工作区文件，不属于 `HEAD`；作者已决定只在本地保存，不上传公开仓库。README 对该未发布文件的链接仍属于文档问题。
- `pyproject.toml` 声明 Apache-2.0，而 README 仍写 planned，且根目录没有 `LICENSE`。DC_STSF 已确认为作者个人未开源模型，应用与模型的许可边界仍需后续处理。
- README 中记录的私有 YRE 演示指标不能仅靠当前仓库复现，因为数据与 Checkpoint 未提交。本次审查没有把它们当作重新验证结果。

## Recent Changes

以下不是机械提交列表，而是近期演化对后续开发的实际影响：

| Commit | Actual impact |
| --- | --- |
| `dc8182b` | 调整 V2.1 配置、Qwen Responses URL 和 Streamlit 配置/对话行为；这是当前审查基线，也是供应商配置与 UI 会话逻辑的最新变更。 |
| `92975ba` | 大幅重排 Streamlit 页面布局；说明 UI 结构近期仍在变化，且仅靠 helper 测试不能证明浏览器显示稳定。 |
| `7463afc` | 把“已外部配准”变成显式输入语义：允许受控警告继续，同时保持系统不做自动配准的边界。 |
| `feefb47`, `8e7684e` | 修复 RAG 评测来源路径在 Windows/Linux 的差异，使 CI 和本地评测更可移植。 |
| `165ccb8` | 增加从任务状态派生的计划、恢复建议、脱敏执行 Trace/报告和 RAG 评测，形成 V2.1 的可观察/可恢复层。 |
| `828b5df` | 引入意图路由、持久任务状态、错误/实验检索并接入 Agent/UI，形成 V2 状态化控制面。 |
| `0cf532d` | 加入本地词法 RAG 基线，不依赖在线 Embedding 或向量数据库。 |
| `d10f99e` | 增加 LangChain Core 工具适配和 OpenCV 回退，但没有把核心 Agent 改为 LangGraph/LangChain Agent。 |
| `2004b83` | 完成 V1 的 H5/TIFF 单 Patch、真实/模拟实验、Streamlit 和结果制品基础，是后续 V2 控制面的执行底座。 |

反复修改最明显的区域是 Streamlit UI、RAG 跨平台路径处理、TIFF 数据契约语义和 Agent 控制面。后续改动这些模块时，应优先查看相邻测试和历史兼容性。

## Next Recommended Development Task

基于当前状态，最合理的下一项工作不是继续增加 Agent 功能，而是完成一个**发布基线与文档一致性任务**：

1. 在 README/发布文档中统一说明：包版本为 `1.1.1`，当前项目开发版本为 V2.1。
2. 明确许可拆分：应用/工具代码是否采用开源许可，以及未开源 DC_STSF 源码和权重如何与公开仓库隔离。
3. 在不改业务逻辑的前提下，同步 README、`docs/data_contract.md`、`docs/llm_agent.md` 及过时 TIFF docstring，使其反映当前工具与工作流，并移除对四份本地私有文档的公开链接。
4. 为当前版本建立可重复验证清单：Ruff、完整 pytest、CLI smoke test、可选私有 GPU/TIFF 测试、手动浏览器 UI 检查，并明确哪些结果不能在 CI 复现。

上述任务直接解决当前仓库事实与文档/发布表述不一致的问题，不引入新产品需求。完成后再由用户选择是优先稳定 UI，还是进入已记录的整景/配准/多模型方向。

## Validation Snapshot

文档生成后应以以下命令重新验证，并把失败原因如实记录：

```powershell
.\.venv\Scripts\python.exe -m ruff check --no-cache src tests examples
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
git diff --check
```

本次审查在 2026-09-17 的实际结果：

- Ruff：`All checks passed!`
- pytest：`87 passed, 95 warnings in 2.39s`
- 警告来自 Rasterio/Affine 的待弃用提示和测试数据的非地理参考提示，没有测试失败。
- `git diff --check`：通过。

这些结果不包含私有 GPU、Checkpoint、真实 LLM 额度或浏览器端到端验证，也不代表远端 GitHub Actions 本次状态已被查询。
