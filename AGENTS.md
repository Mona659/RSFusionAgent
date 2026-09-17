# RSFusionAgent Development Guide

本文件面向后续 Codex/开发者，说明在本仓库中应如何安全、最小化地开展开发。项目事实与当前状态分别见 `docs/PROJECT_CONTEXT.md` 和 `docs/DEVELOPMENT_STATUS.md`；实现细节见 `docs/ARCHITECTURE.md`；既有技术决策见 `docs/DECISIONS.md`。

## 1. Project Overview

RSFusionAgent 是一个本地优先的遥感时空谱融合实验 Agent。当前实现支持 H5 Patch 与原始 GeoTIFF 两条实验路径，以受约束的 LLM Function Calling 循环编排输入检查、裁剪准备、独立模型环境预检、单 Patch 推理、指标计算、结果可视化、知识检索和运行记录。

项目的关键安全边界是：数据、Checkpoint 和模型推理均保留在本机；LLM 只接收结构化摘要与工具返回，不应获得原始影像、模型权重、API Key 或任意本地路径访问能力。

## 2. Repository Structure

```text
RSFusionAgent/
├── src/rsfusion_agent/
│   ├── agent/       # Agent 循环、工具箱、意图路由、状态、计划、可观测性
│   ├── models/      # DC_STSF 模型结构
│   ├── rag/         # 本地词法检索、错误检索、实验历史检索、RAG 评测
│   ├── runtime/     # 独立 PyTorch 进程中的环境预检与模型推理入口
│   ├── tools/       # H5/TIFF、裁剪、指标、制品、运行时桥接等确定性工具
│   ├── ui/          # Streamlit UI
│   └── cli.py       # 所有 CLI 子命令的统一入口
├── knowledge/       # 受控本地知识库与 RAG 评测集
├── tests/           # 单元/集成级自动化测试
├── docs/            # 使用、架构、状态和历史文档
├── scripts/         # 离线恢复等辅助脚本
├── examples/        # 可公开运行的小型示例
├── .github/workflows/ci.yml
├── pyproject.toml
└── README.md
```

`data/`、绝大多数 `outputs/`、本地环境与私有 Checkpoint 均被 Git 忽略，不能假定它们在其他机器或 CI 中存在。

## 3. Development Principles

1. 修改前先阅读当前实现、相邻测试及相关文档，不依赖旧会话记忆推断现状。
2. 开始修改前，先列出计划修改的文件及每个文件的目的。
3. 优先修改与当前任务直接相关的最小范围代码。
4. 不要为完成一个小需求进行无关重构、改名或目录迁移。
5. 不要擅自改变已经确定的架构、安全边界、文件格式或 CLI 接口。
6. 如果需求与现有架构冲突，先明确指出冲突及影响，再请求决策。
7. 不确定时不要猜测；从代码、测试、配置或 Git 历史无法确认的内容应标为“需要确认”。
8. 不要在代码、命令、日志、文档或测试夹具中写入 API Key、私有数据路径或 Checkpoint 内容。
9. 不要提交 `data/`、真实遥感数据、模型权重或本地 `outputs/` 运行制品。
10. 修改后必须检查 `git diff`，避免混入用户原有改动或无关格式化。

## 4. Architecture Constraints

- **本地优先**：原始 H5/TIFF、Checkpoint、模型张量和生成制品不得发送给 LLM。
- **模型环境隔离**：Agent/UI 环境不要求安装 PyTorch；GPU 预检与推理由 `tools/model_runtime.py` 调用配置的模型 Python 子进程完成。
- **工具白名单**：LLM 只能调用相应 `QueryToolbox`、`AgentToolbox` 或 `TiffAgentToolbox` 暴露的工具。不得引入任意 Shell、任意 Python、任意文件读取工具。
- **本地配置注入**：可信路径和运行配置由上下文对象提供，不得改为 LLM 工具参数。
- **显式推理授权**：融合推理必须由用户请求中的明确意图授权；不得仅依据模型自行判断启动昂贵推理。
- **TIFF 清单授权**：原始 TIFF 推理必须消费已生成并检查过的 `crop_manifest.json`，不得绕过像素对应关系清单。
- **有界顺序调用**：当前 LLM 循环有最大轮数，并关闭并行工具调用。变更轮次、并发或重试语义前应补充测试并说明成本/安全影响。
- **结果防陈旧**：UI/CLI 不应把上一轮旧的 `agent_result.json` 当成本轮成功结果。
- **可观测性脱敏**：Trace 可以记录工具名、状态、耗时和参数字段名，但不能泄漏绝对路径、API Key、原始数据或权重。
- **当前不是 LangGraph 架构**：不要在未有明确需求的情况下引入 LangGraph。现有 LangChain 仅为 `StructuredTool` 适配层。
- **当前不是向量 RAG**：现有 RAG 为本地词法检索，不依赖 Embedding 或向量数据库。
- **模型许可边界**：DC_STSF 是项目作者个人持有、尚未开源的算法模型。不得擅自把模型源码、权重或衍生实现声明为 Apache-2.0 等开源许可。

## 5. Important Files

| Path | Responsibility |
| --- | --- |
| `src/rsfusion_agent/cli.py` | CLI 参数、上下文组装、Agent/工作流入口、结果写入 |
| `src/rsfusion_agent/agent/llm_workflow.py` | 有界 Function Calling Agent 循环与系统提示词 |
| `src/rsfusion_agent/agent/llm_tools.py` | 三类工具箱、工具 Schema、前置条件和本地授权边界 |
| `src/rsfusion_agent/agent/intent_router.py` | 规则优先、LLM 回退的意图识别 |
| `src/rsfusion_agent/agent/task_state.py` | 持久化轻量任务阶段与已完成工具集合 |
| `src/rsfusion_agent/agent/task_plan.py` | UI/可观测性使用的执行计划和恢复建议 |
| `src/rsfusion_agent/agent/observability.py` | 脱敏执行 Trace 与 Markdown 报告 |
| `src/rsfusion_agent/agent/workflow.py` | H5 单 Patch 确定性融合工作流 |
| `src/rsfusion_agent/agent/tiff_workflow.py` | TIFF 清单驱动的真实/模拟实验工作流 |
| `src/rsfusion_agent/tools/model_runtime.py` | Agent 环境到独立模型环境的子进程桥接 |
| `src/rsfusion_agent/runtime/*.py` | PyTorch/CUDA 环境预检和实际推理进程 |
| `src/rsfusion_agent/rag/` | 本地知识、错误、历史检索与 RAG 评测 |
| `src/rsfusion_agent/ui/streamlit_app.py` | Streamlit 配置、会话、子进程执行和结果渲染 |
| `knowledge/` | 受控知识源与 RAG 评测用例 |
| `tests/` | 当前行为契约的主要可执行证据 |
| `pyproject.toml` | 依赖、可选 extras、入口点、Ruff 配置和包版本 |
| `.github/workflows/ci.yml` | Python 3.10 下 Ruff 与 pytest 的 CI 基线 |

## 6. Coding Conventions

- Python 最低版本为 3.10；使用类型注解和 `pathlib.Path`。
- 结构化输入/输出使用 Pydantic v2 模型；外部边界优先输出 JSON 可序列化数据。
- Ruff 配置以 `pyproject.toml` 为准：行宽 100，目标 Python 3.10，检查 `E4/E7/E9/F/I`。
- `src/rsfusion_agent/models/dc_stsf.py` 当前被 Ruff 排除；不要借小需求重排或格式化该研究模型文件。
- 工具实现应保持确定性；授权、路径限制和前置条件在工具箱中执行，而不是依赖 Prompt。
- 新工具必须同步考虑：工具 Schema、工具箱 allowlist、LangChain 适配（如适用）、状态/计划映射、可观测性和测试。
- 面向用户的错误应经过分类/脱敏；底层诊断可以保存在本地制品中，但不得暴露密钥。
- CLI 参数保持向后兼容。若必须变更，先说明迁移方案并更新 README、测试和相关文档。

## 7. Testing Requirements

- 每次代码修改至少运行与改动模块直接相关的测试。
- 涉及 Agent、工具 Schema、状态、CLI 或公共数据模型时，应运行完整测试套件。
- 涉及导入、格式或 CI 时，应同时运行 Ruff。
- UI 改动至少运行 `tests/test_streamlit_ui.py`，但要明确它只覆盖辅助逻辑，不等价于真实浏览器/响应式布局测试。
- GPU/Checkpoint/TIFF 私有数据端到端测试依赖本地资源；无法运行时必须明确说明未验证的边界。
- 不得为了让测试通过而删除断言、降低关键前置条件或伪造实际推理成功。

## 8. How to Run

PowerShell 示例：

```powershell
# 开发环境（Agent/UI；不等于 PyTorch 模型环境）
.\.venv\Scripts\python.exe -m pip install -e ".[dev,llm,langchain,ui]"

# 查看 CLI
.\.venv\Scripts\python.exe -m rsfusion_agent.cli -h

# 启动 Streamlit
.\.venv\Scripts\python.exe -m streamlit run src/rsfusion_agent/ui/streamlit_app.py

# 公开的轻量栅格示例
.\.venv\Scripts\python.exe examples/create_demo_raster.py
```

LLM API Key 只能通过环境变量配置，例如 `RSFUSION_LLM_API_KEY`、`DASHSCOPE_API_KEY` 或 `OPENAI_API_KEY`。不要把密钥作为 CLI 参数。模型推理需另行配置包含 PyTorch/模型依赖的 Python 可执行文件。

## 9. How to Validate Changes

```powershell
# 静态检查
.\.venv\Scripts\python.exe -m ruff check --no-cache src tests examples

# 完整测试；禁用 pytest cache 可避免受限目录的缓存警告
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider

# 检查只包含预期改动
git status --short
git diff --check
git diff --stat
git diff
```

文档改动还应核对：文中路径存在、类/函数名可由 `rg` 找到、命令与 `cli.py`/`pyproject.toml` 一致、计划功能没有写成已完成功能。

## 10. Files/Modules That Should Not Be Modified Casually

- `src/rsfusion_agent/models/dc_stsf.py`：研究模型结构和 Checkpoint 兼容性敏感。
- `src/rsfusion_agent/runtime/yre151_runner.py`：张量形状、波段数、缩放和权重加载构成模型契约。
- `src/rsfusion_agent/tools/h5_patch.py`：YRE151 H5 通道布局和归一化契约。
- `src/rsfusion_agent/tools/tiff_*.py` 与 `agent/tiff_workflow.py`：地理元数据、裁剪窗口、清单和真实/模拟实验语义。
- `src/rsfusion_agent/agent/llm_tools.py`：工具 allowlist、路径隔离和执行前置条件。
- `src/rsfusion_agent/agent/task_state.py`：持久化阶段值可能已存在于本地历史文件中。
- `src/rsfusion_agent/agent/llm_state.py`、`agent/state.py`：JSON 制品和 UI/CLI 消费的公共 Schema。
- `src/rsfusion_agent/cli.py`：公开命令及参数兼容性。
- `knowledge/evaluations/rag_eval.json`：RAG 回归基线；变更必须说明评测含义。
- `.github/workflows/ci.yml`、`pyproject.toml`：CI/依赖基线。

## 11. Known Constraints

- 仓库不包含私有数据、Checkpoint 或可复现完整 GPU 推理所需的模型环境。
- 原始 TIFF 假定数据已外部配准；`external_registration` 只把部分元数据差异降级为警告，不执行自动配准、重投影或像素残差验证。
- 当前只执行选定的非重叠 Patch，不是全景拼接流水线。
- RAG 为离线词法召回，不是语义向量检索；评测衡量来源命中率，不衡量最终答案正确性。
- Streamlit 对话保存在当前 Session State，服务器/浏览器会话重启后不保证保留。
- CI 不具备 GPU、私有数据和 Checkpoint，因此不能替代本地模型端到端验证。
- UI 的真实浏览器响应式布局当前没有自动化端到端覆盖。

## 12. Rules for Future Codex Sessions

1. 首先阅读本文件及 `docs/PROJECT_CONTEXT.md`、`docs/ARCHITECTURE.md`、`docs/DEVELOPMENT_STATUS.md`、`docs/DECISIONS.md`。
2. 再运行 `git status --short`，确认用户已有修改；不要覆盖或回滚未知改动。
3. 以当前代码和测试为最高事实来源；README/历史总结若与代码冲突，应指出而不是照抄。
4. 在动手前向用户给出简短实施计划和拟修改文件。
5. 只修改任务需要的文件；不要顺带解决无关问题。
6. 修改后检查完整 `git diff`，并运行可行的相关测试与 Ruff。
7. 如测试依赖网络、GPU、私有数据、API 额度或 Checkpoint而无法运行，准确列出未验证项。
8. 不要自行提交、推送、删除文件、重写历史或更换架构，除非用户明确要求。
9. 不要把“拟开发”“历史文档宣称”或“推测”写成当前已实现事实。
10. 更新行为时同步更新最接近该行为的测试和文档，但不要机械复制五份上下文文档的内容。
11. `docs/v1_development_summary.md`、`docs/v2_development_summary.md`、`docs/v2_1_development_plan.md`、`docs/v2_five_day_plan.md` 是作者本地开发计划/总结，不得加入提交或上传到远端公开仓库。
