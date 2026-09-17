# Architecture Decisions

本文件记录从当前代码、测试、已跟踪文档和提交历史中能够确认的技术决策。若只能确认“现在如何实现”而无法确认最初动机，会明确写出 **Historical reason not confirmed.**

## D-001 — Keep data and model execution local

**Decision**  
原始遥感数据、模型权重、张量和推理结果在本地处理；LLM 仅接收请求、工具 Schema 和结构化摘要。

**Context**  
项目处理本地 H5/GeoTIFF、私有 Checkpoint 和 GPU 推理。将这些内容直接交给外部模型会扩大隐私、带宽和执行风险。

**Reason**  
README、系统提示词、工具上下文和测试共同把“不上传 H5/权重、不向模型暴露任意路径”作为安全边界。

**Current Implementation**  
工具箱由本地上下文注入路径；LLM 工具参数不接受任意路径。`LLMFusionAgent` 只转发工具生成的文本/JSON 摘要。API Key 由环境变量读取。

**Constraint**  
新增工具不得返回原始数组、Checkpoint 内容、密钥或不必要的绝对路径。远程执行方案会改变核心安全模型，不能作为小改动引入。

## D-002 — Use a custom bounded Function Calling loop as the core Agent

**Decision**  
核心 Agent 使用 `LLMFusionAgent` 的自定义顺序 Function Calling 循环，而不是 LangGraph 或通用自主 Agent 运行时。

**Context**  
融合工具包含本地文件和 GPU 副作用，需要限制轮数、顺序、工具集合以及推理授权。

**Reason**  
当前实现让轮次、错误捕获、调用 ID、工具 Trace、并行开关和显式推理授权全部可直接审计。Git 历史显示 Agent 循环先于 LangChain 适配存在。

**Current Implementation**  
`agent/llm_workflow.py` 设置最大轮数，`parallel_tool_calls=False`，逐个执行工具并把 `function_call_output` 回传。工具失败不会自动开放其他执行能力。

**Constraint**  
不能在没有迁移设计和等价安全测试的情况下，用 LangGraph/LangChain Agent 替换该循环。为什么最初没有采用 LangGraph：**Historical reason not confirmed.** 当前仓库也没有 LangGraph 依赖。

## D-003 — Treat LangChain as an optional adapter, not an execution authority

**Decision**  
LangChain Core 仅用于把既有工具包装成 `StructuredTool`；授权和业务执行继续由工具箱控制。

**Context**  
项目需要展示标准工具接口兼容性，同时避免形成两套执行规则。

**Reason**  
`langchain_tools.py` 将调用委派给同一工具箱，`langchain_agent.py` 是确定性 façade。提交历史也把它描述为 adapter。

**Current Implementation**  
`langchain-core` 是可选 extra。未安装时基础 CLI/工作流仍可使用。

**Constraint**  
LangChain 包装不得绕过 allowlist、路径注入、前置条件和显式推理授权。工具 Schema 变更需同步验证包装层。

## D-004 — Enforce tools with code-level allowlists and trusted contexts

**Decision**  
按任务类型使用 `QueryToolbox`、`AgentToolbox`、`TiffAgentToolbox` 三个工具集合；可信路径/配置放在上下文中，而不是让 LLM 提供。

**Context**  
只读问答、H5 实验和 TIFF 实验的权限与前置条件不同。

**Reason**  
代码级隔离比仅依赖 Prompt 更可验证，也能在 LLM 产生错误工具调用时安全失败。

**Current Implementation**  
每个工具箱暴露固定 Schema 和分发逻辑。融合工具检查用户原始请求授权；TIFF 工具检查原始输入、裁剪清单等前置状态。

**Constraint**  
禁止新增任意 Shell、任意 Python 或任意文件读取工具。新增工具时必须同步 Schema、分发、状态/计划、可观测性和测试。

## D-005 — Separate the Agent environment from the PyTorch model environment

**Decision**  
Agent/UI Python 不承担 PyTorch 模型加载；预检和推理由用户配置的模型 Python 子进程完成。

**Context**  
Windows/CUDA/PyTorch/研究模型依赖较重，且 UI/Agent/RAG 本身不需要 PyTorch。

**Reason**  
README 和 `model_runtime.py` 明确强调运行环境隔离；这样可避免轻量 Agent 环境被模型依赖绑定，也能将原生崩溃隔离为子进程失败。

**Current Implementation**  
桥接层以参数列表启动 `preflight_runner`/`yre151_runner`，注入 `PYTHONPATH`，施加超时并解析最后一行 JSON。

**Constraint**  
不要把 PyTorch 加入基础 Agent 依赖来规避模型环境配置。协议变更需同时更新两个 runner、桥接层和测试。

## D-006 — Keep inference tied to the built-in YRE/DC_STSF contract

**Decision**  
当前模型运行时固定支持 DC_STSF 的 4 波段 MS、151 波段 HS、3 倍空间尺度契约。

**Context**  
H5 通道布局、TIFF Patch 构造、模型结构和 Checkpoint 形状相互耦合。

**Reason**  
代码和 README 都将替换模型描述为需要新适配器与数据/制品验证，而不是只换 Checkpoint。作者已确认 DC_STSF 是其个人算法模型，目前尚未开源。

**Current Implementation**  
`h5_patch.py`、`yre151_runner.py` 和 `models/dc_stsf.py` 共同实施该契约，并严格加载状态字典。

**Constraint**  
不得通过放宽形状检查或非严格权重加载伪装“多模型支持”。新模型需要明确的输入适配、运行时、结果和测试契约。不得因应用工程中出现 Apache-2.0 元数据，就推断 DC_STSF 模型源码或权重也已获得开源授权。

## D-007 — Make the TIFF crop manifest an explicit inference authorization artifact

**Decision**  
原始 TIFF 融合必须基于已生成、可追踪且已检查的 `crop_manifest.json`。

**Context**  
不同分辨率影像之间的像素窗口对应关系不能由模型或隐式文件顺序猜测。

**Reason**  
已跟踪 V1 发布说明明确把清单定义为源像素对应指令，而不是自动配准声明。

**Current Implementation**  
先检查原始输入，再由裁剪工具写入窗口/Patch 信息；`TiffAgentToolbox` 在融合前要求清单存在并经过检查，`YRE151TiffPatchAgent` 消费该清单。

**Constraint**  
不得让 LLM 直接构造任意裁剪路径或跳过清单。清单 Schema 的不兼容变更需要迁移既有本地运行记录。

## D-008 — Distinguish strict metadata alignment from externally registered inputs

**Decision**  
TIFF 检查支持严格元数据模式和 `external_registration` 模式。后者可把特定 CRS/bounds/resolution 差异降级为警告，但不执行配准或重投影。

**Context**  
实际数据可能已由外部流程配准，但元数据仍有小差异；把所有差异判为硬阻塞过于绝对。

**Reason**  
提交 `7463afc` 和对应测试明确引入该语义，以保留用户对外部配准关系的责任，同时避免系统作出未经验证的对齐承诺。

**Current Implementation**  
检查结果区分 blocking issues 与 warnings；后续按显式像素窗口裁剪，不自动更改坐标或像素网格。

**Constraint**  
UI/回答必须说明它是用户声明，不是残余偏差验证。自动配准若未来实现，应成为独立、可度量的步骤。

## D-009 — Persist minimal task state and derive plans from tool history

**Decision**  
持久状态只保存阶段与已完成工具；执行计划由该状态、输入模式和清单可用性派生，而不是从聊天文本恢复。

**Context**  
UI/后续请求需要知道任务进度，但不能持久化影像数组、权重或密钥。

**Reason**  
V2/V2.1 提交明确增加轻量状态、计划和恢复层。基于工具记录比解析自然语言历史更确定。

**Current Implementation**  
`TaskState` 原子写入 `task_state.json`；`build_task_execution_plan()` 从 `completed_tools` 计算 ready/blocked/completed 步骤。

**Constraint**  
阶段字符串和已完成工具名可能已进入本地历史文件，改名需要兼容处理。当前单一 stage 只是摘要，依赖判断必须继续参考 `completed_tools`。

## D-010 — Route explicit intents with rules before spending an LLM call

**Decision**  
对明确状态、检查、裁剪、融合等请求先使用规则；只有不明确时才请求 LLM 进行 Schema 约束分类。

**Context**  
常见命令应低成本、可预测，且外部 LLM 可能无额度或暂时不可用。

**Reason**  
当前路由器和测试体现“规则优先、LLM 回退”；失败时回退到只读知识意图，避免意外执行。

**Current Implementation**  
`route_intent()` 产生 `IntentDecision`，记录来源和置信度；UI 使用结果选择流程。

**Constraint**  
新增规则不能让普通问答误触发融合。涉及副作用的意图仍需工具箱授权，不能只信路由结果。

## D-011 — Use local lexical RAG without embeddings or a vector database

**Decision**  
当前知识检索采用本地文件、字符切块和词法覆盖率评分。

**Context**  
项目需要离线、轻量、无需额外服务的知识/错误检索，并要求来源可见。

**Reason**  
提交 `0cf532d` 将其作为本地 RAG 基线；代码没有 Embedding 或向量数据库依赖。为什么未选择特定向量库：**Historical reason not confirmed.**

**Current Implementation**  
读取 Markdown/JSON，排除评测目录，返回 top-k 片段与来源；错误与历史实验使用受限目录/摘要检索。

**Constraint**  
不能把当前能力描述为语义向量检索。引入 Embedding 会增加模型、索引、版本、隐私和评测决策，应单独设计。

## D-012 — Evaluate retrieval by expected source hits

**Decision**  
当前 RAG 回归评测以期望来源是否出现在 top-k 为指标。

**Context**  
词法检索首先需要跨平台、确定性的来源召回基线。

**Reason**  
V2.1 增加版本化评测集，随后的修复专门处理 Windows/POSIX 路径差异。

**Current Implementation**  
`rag/evaluation.py` 规范化来源分隔符并计算 case/source hit rate；`knowledge/evaluations/` 不进入检索语料。

**Constraint**  
来源命中不等于最终答案正确。文档和简历不得把该指标表述为回答准确率。

## D-013 — Use OpenAI-compatible Responses API with environment-only secrets

**Decision**  
LLM 通过统一 Responses API 客户端访问 OpenAI、Qwen、DeepSeek 或 Custom 兼容端点；API Key 不作为 CLI 参数。

**Context**  
需要多供应商切换，同时避免密钥出现在命令历史、进程参数和 UI 持久配置中。

**Reason**  
`llm_client.py`、CLI 错误信息和 README 明确要求环境变量。Qwen URL 归一化解决其 Chat-compatible 与 Responses 路径差异。

**Current Implementation**  
优先读取 `RSFUSION_LLM_API_KEY`，并支持供应商环境变量回退；`store=False`。UI 只持久化非秘密偏好。

**Constraint**  
不得添加 API Key 文本框持久化或 CLI `--api-key`。供应商模型 ID、额度和价格属于外部状态，代码默认值不能当作永久有效事实。

## D-014 — Produce structured, sanitized local observability artifacts

**Decision**  
Agent 运行应同时产生机器可读 Trace 和人类可读报告，并限制其中的敏感信息。

**Context**  
本地科研流程需要定位工具失败、恢复任务和解释结果，又不能把路径/数据泄漏到共享日志。

**Reason**  
提交 `165ccb8` 明确增加 observable and recoverable execution。

**Current Implementation**  
`observability.py` 记录事件、耗时、状态、参数字段名、Token/成本、计划和制品 basename，输出 `execution_trace.json` 与 `agent_execution_report.md`。

**Constraint**  
新增 Trace 字段必须经过脱敏审查。不要直接序列化工具上下文、完整参数值、原始异常环境或影像内容。

## D-015 — Use categorized errors and bounded, narrow retries

**Decision**  
错误按可操作类别解释；自动重试仅用于已识别的 Windows 原生 fast-fail，且次数有上限。

**Context**  
数据/配置/CUDA 错误盲目重试无效，原生偶发崩溃则可能在有限重试后恢复。

**Reason**  
V1 发布说明和代码均明确限制重试范围，所有尝试进入运行结果。

**Current Implementation**  
`error_diagnosis.py` 给出类别、摘要和行动建议；`model_runtime.py` 只识别特定退出码，最多允许少量额外尝试。

**Constraint**  
不得扩大为“任何失败自动重试”。新重试策略必须有明确的可重试错误集合、上限、Trace 和测试。

## D-016 — Treat CLI, JSON artifacts and persisted state as compatibility surfaces

**Decision**  
CLI 子命令/参数、Pydantic 结果模型、`agent_result.json`、`run_manifest.json`、`crop_manifest.json` 和 `task_state.json` 都是当前系统的跨模块接口。

**Context**  
Streamlit、恢复脚本、历史运行读取和测试依赖这些边界，即使项目没有 HTTP API。

**Reason**  
代码中存在多个生产者/消费者，历史结果会在进程外长期存在。完整的最初接口版本政策：**Historical reason not confirmed.**

**Current Implementation**  
CLI 统一写入结构化 JSON，UI 以只读方式载入历史结果，恢复脚本读取 `run_manifest.json`。

**Constraint**  
字段删除、重命名或语义变化必须检查所有消费者，并为既有本地文件提供兼容读取或清晰迁移说明。

## D-017 — Split the application license from the proprietary model license

**Decision**
应用、Agent、工具、RAG、UI、测试和文档代码采用根目录 Apache License 2.0。`src/rsfusion_agent/models/dc_stsf.py` 以及任何 DC_STSF 权重、Checkpoint、转换权重或权重衍生物明确排除在该授权范围之外，由 `MODEL_LICENSE.md` 保留全部权利。

**Context**
公开仓库需要明确的应用工程许可证，但 DC_STSF 是作者个人持有、尚未开源的算法模型。原有 `pyproject.toml` 的笼统 Apache-2.0 元数据、README 的 planned 表述和缺失的根许可证会让应用代码与模型代码的权利边界不清晰。

**Reason**
拆分许可既允许招聘展示和社区复用应用工程部分，也避免公开许可证被错误解释为对模型结构和权重的授权。作者已明确确认这一许可策略。

**Current Implementation**
根目录 `LICENSE` 提供 Apache-2.0 正文；`MODEL_LICENSE.md` 列出排除材料；README、`pyproject.toml` 和开发规范同步说明该边界。公开仓库不包含模型权重或私有 YRE 数据。

**Constraint**
新增模型实现或权重时必须明确其许可归属。不得删除模型排除说明、把 DC_STSF 文件纳入 Apache-2.0，或在未取得书面许可时使用、修改、再分发模型材料。未来打包或发布 Python 分发物时必须保留两份许可证说明。

## Items Requiring Confirmation

已确认：Python 包正式版本为 `1.1.1`，当前项目开发版本为 V2.1；四份未跟踪开发计划/总结只保存在本地；“长期记忆”仅指当前本地历史实验检索；DC_STSF 是作者个人算法且尚未开源。

仍需确认：

1. `TaskStage.INFERENCE_AUTHORIZED`、`PlanStepStatus.NOT_REQUIRED` 是计划接入的保留接口，还是可在兼容迁移后移除。
