# A/B/C 全量运行、断点续跑与 OpenRouter

本页对应本地 `chartsandbox` 流程。发布数据集自带的 macOS Seatbelt runner 是另一套入口，本页不依赖它。以下命令从仓库的 `sandbox/` 目录执行。

## 1. 环境与数据

```bash
uv sync --frozen
```

需要另行准备 ChartRepairBench 发布数据目录，包含 `data/A.jsonl`、`data/B.jsonl`、`data/C.jsonl`、`instances/` 和 `scoring/`。数据不会随本仓库推送。

在当前 ChartNet 工作区，路径为：

```bash
export CRB_RELEASE=/home/tuo96248/projects/ChartNet/data/chartrepairbench-v1.3
export CRB_TASKS=/home/tuo96248/projects/ChartNet/data/chartrepairbench-tasks-full
export CRB_OUTPUT=/home/tuo96248/projects/ChartNet/data/runs/openrouter-full
```

其他机器请替换为相应绝对路径。不要将新输出指向已有的 DeepSeek 历史实验目录。

## 2. 构建三个 Track

```bash
.venv/bin/python scripts/prepare_release.py \
  --release "$CRB_RELEASE" --out "$CRB_TASKS"
```

默认构建 A/B/C 全部任务，发布数据中分别为 1,900 / 1,900 / 2,200 题。生成结构：

```text
$CRB_TASKS/
├── trackA/<task_id>/task.yaml + workspace/ + oracle/
├── trackB/<task_id>/task.yaml + workspace/ + oracle/
└── trackC/<task_id>/task.yaml + workspace/ + oracle/
```

- 默认任务预算为 6 轮、14 步、1,800 秒，与此前本地转换脚本默认值一致。
- 若要使用不限制总轮数/步数/时间的任务，构建时增加 `--uncapped`；使用独立任务和输出目录。运行时还要指定 `--episode-timeout 0` 才会关闭外层进程超时。每次 Python 执行和 HTTP 请求仍有超时。
- `--track A` 可只构建一轨；`--n 10` 可先构建每轨前 10 题。
- 再次运行会跳过源行与预算相同的已构建任务，移除 `--n` 即可扩展至全量。任务通过临时目录完成后整体重命名，避免把半成品当成任务。
- `--rows <sample-dir>` 可读取另一个含 `data/A.jsonl` 等文件的抽样目录，图像仍从 `--release` 获取。
- 不会删除已有任务；遇到不同配置或来源会报错，要求使用新目录。

验证任务契约和输入文件，不调用模型：

```bash
.venv/bin/python scripts/run_suite.py "$CRB_TASKS" \
  --model openrouter:vendor/model-slug --out "$CRB_OUTPUT" --validate-only
```

`--validate-only` 验证任务，不验证远程模型是否存在或兼容。

## 3. OpenRouter 设置

只需现有 `openai` SDK，不需要 OpenRouter 专用 SDK。

- 环境变量：`OPENROUTER_API_KEY`。
- 默认 API base URL：`https://openrouter.ai/api/v1`。
- 模型写法：`openrouter:<OpenRouter 模型 slug>`，例如 `openrouter:vendor/model-slug`。这里的占位符必须替换为平台上实际存在的 ID。
- Agent 必须支持图像输入；Judge 还必须支持 `json_schema` structured outputs；使用 `--protocol tools` 时 Agent 还需要支持 tools。
- 不要假定原有直连接口的 `deepseek-v4-flash-vision-exp` 是 OpenRouter 上可用的同名模型。模型 ID 以 OpenRouter 模型列表为准。
- 批量入口默认 `--effort default --judge-effort default`，不发送额外 reasoning effort。需要指定时使用模型支持的 `--effort high` 等选项。
- 模型回复的 reasoning details 会在后续对话中原样回传；实际返回的文本 reasoning、响应 ID、模型、供应商和 usage 会记录到统计中。
- 请求设置 `provider.require_parameters=true`，避免供应商悄悄忽略工具、结构化输出或显式 reasoning 参数。没有配置跨模型 fallback；同一模型仍可能由平台路由至不同供应商，响应 metadata 在可用时保留。
- Judge 不支持结构化输出时，不会退回 vLLM 专用参数。该请求错误会被标记为评估异常。
- 输入/输出/缓存/reasoning Token 分开记录；reasoning 不在输出之外重复相加。平台返回的实际 `usage.cost` 保存在 `agent_stats.responses` / `judge_stats.responses`，不冒充本地价格表估算的 `cost_usd`。

在终端安全输入密钥（不回显）：

```bash
read -rsp 'OpenRouter API key: ' OPENROUTER_API_KEY
export OPENROUTER_API_KEY
```

设置真实模型 ID：

```bash
export CRB_AGENT_MODEL='openrouter:vendor/model-slug'
export CRB_JUDGE_MODEL='openrouter:vendor/judge-model-slug'
```

Agent 和 Judge 可使用不同模型。若使用同一模型，请在实验说明中注明。

## 4. 启动与续跑

先使用 `--limit 3` 检查接口和少量任务；这是按排序选取的前三题，不是每轨三题：

```bash
.venv/bin/python scripts/run_suite.py "$CRB_TASKS" \
  --model "$CRB_AGENT_MODEL" --judge "$CRB_JUDGE_MODEL" \
  --out "$CRB_OUTPUT" --workers 2 --limit 3
```

确认后移除 `--limit`，使用相同输入、模型配置与输出目录：

```bash
.venv/bin/python scripts/run_suite.py "$CRB_TASKS" \
  --model "$CRB_AGENT_MODEL" --judge "$CRB_JUDGE_MODEL" \
  --out "$CRB_OUTPUT" --workers 6
```

输出结构自动保留 Track，避免 A/B 同名任务互相覆盖：

```text
$CRB_OUTPUT/
├── summary.json
├── trackA/<task_id>/
├── trackB/<task_id>/
├── trackC/<task_id>/
└── .attempts/              # 中断或评估异常尝试的归档
```

也支持只传 `$CRB_TASKS/trackB` 并输出到 `$CRB_OUTPUT/trackB`。同一运行系列保持输入/输出的目录层级一致。

### 续跑保证

- 默认自动续跑，`--resume` 仍可使用，但不再是必需参数。
- 每题完成立即原子写入 `result.json`；每题返回后立即原子更新 `summary.json`，失败也更新。
- 每一步执行后追加并刷盘 `trajectory.jsonl`；执行前保存 Agent checkpoint，整题未结束也能检查已产生的轨迹。
- 是否完成依据可解析的结果、正确 task ID、轨迹行数与 `steps_taken` 一致、workspace 存在、且无 evaluator error。**模型做错、零分或耗尽任务预算也属于完成结果，不会因低分自动重跑。**
- 任务内容哈希、运行参数、沙箱源代码哈希和记录的依赖版本必须一致。更换模型、Judge、预算、任务内容或沙箱实现，请使用新的输出目录。
- `--workers` 和 `--limit` 不影响运行身份，可以先跑子集，再提高并发、移除数量限制继续。
- 损坏/缺失结果、进程超时或 evaluator error 的尝试会整体归档，下一次从该题重新开始。已有无 manifest 的旧实验不会被自动接管或覆盖。
- **不恢复题内 Python 内核状态**：中断时正在运行的题重新开始；之前已完成的题跳过。
- Ctrl-C 停止子进程组；下次执行同一命令即可继续。一个输出目录同时只允许一个批量控制进程。
- 汇总 Token 仅包含完成任务；失败尝试的 checkpoint、日志和结果保存在 `.attempts/`，分析实际总开销时应另外计入。
- 未配置 `--judge` 时批量入口拒绝启动，避免把 mock 评分混入正式实验；只有显式 `--mock-judge` 才能进行无真实评分的流程测试。

## 5. 保存哪些内容、如何计算指标

每题保存：

| 文件 | 用途 |
|---|---|
| `result.json` | 子目标评分、停止原因、轮数、Token、耗时、Agent transcript、Judge 统计 |
| `trajectory.jsonl` | 每步工具参数（包括执行代码）、stdout/stderr/错误等摘要 |
| `renders/<step>/` | 各步已写出、自动捕获或显式查看的图像版本，避免后续覆盖丢失 |
| `agent_config.json` | 系统提示和初始观察 |
| `agent_checkpoint.json` | 最近已返回的 Agent 状态与即将执行的动作；非完整内核快照 |
| `workspace/chart.py` | 最终提交脚本；规则评分的输入 |
| `workspace/output.png` | 最终图像；视觉评分的输入，失败任务可能缺失 |
| `task.yaml` / `task_config.json` | 任务协议及发布实例对应关系，位于 Agent workspace 之外 |
| `run_manifest.json` | 内容哈希和运行配置，供续跑校验 |
| `process.log` / `failure.json` | 子进程输出及基础设施失败说明 |

**仅有 traces 不足以重算全部规则指标**：还需要最终脚本和原始发布包的 reference/table/summary。请保留发布包版本与任务目录。图像恢复与脚本重跑是不同评分对象，`restored_renders` 会明确标记回退。

依次为各轨补充规则分数（重新执行代码，不调用模型；默认串行）：

```bash
for track in A B C; do
  .venv/bin/python scripts/score_rules.py "$CRB_OUTPUT/track$track" \
    --release "$CRB_RELEASE" --tasks "$CRB_TASKS/track$track"
done
```

该脚本调用发布包的原生评分器，原子写回每题 `rule_based`，不重新实现评分规则。

只读汇总：

```bash
.venv/bin/python scripts/summarize_metrics.py "$CRB_OUTPUT" > "$CRB_OUTPUT/metrics.json"
```

规则 Executability / Data Fidelity / Recovery / Preservation 与 Judge repair / visual quality 分列，附每项有效样本数；不以 Judge 分数填补规则缺失项，不把未定义值记成零。正确图表的 Recovery 不适用。真实脚本执行失败的规则零分保留；mock 与 Judge 异常不充当真实 Judge 分数。

## 6. 离线检查

```bash
.venv/bin/python scripts/dry_run_agent.py
.venv/bin/python scripts/dry_run_tagged.py
.venv/bin/python scripts/dry_run_sandbox.py
.venv/bin/python scripts/dry_run_openrouter.py
.venv/bin/python scripts/dry_run_suite.py
.venv/bin/python scripts/dry_run_release.py --release "$CRB_RELEASE"
```

OpenRouter 检查使用假 SDK 客户端；suite 检查使用真实子进程和 scripted agent；release 检查使用正确参考代码回放，覆盖三个 Track、三种绘图库及 C 的正确图表。它们验证运行与评分链路，不测量模型能力，不调用真实 LLM。设置密钥后仍需用实际选定模型进行小规模在线验证。

## OpenRouter 官方依据

- [模型列表与能力字段](https://openrouter.ai/docs/guides/overview/models)
- [图像输入](https://openrouter.ai/docs/guides/overview/multimodal/image-understanding)
- [Quickstart / OpenAI SDK 接入](https://openrouter.ai/docs/quickstart)
- [Reasoning 与 reasoning_details 回传](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [Structured Outputs](https://openrouter.ai/docs/guides/features/structured-outputs)
- [Provider Routing](https://openrouter.ai/docs/guides/routing/provider-selection)
- [Usage Accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting)

文档核对日期：2026-09-15。
