# Lens

> 把「这次改动感觉变好了」变成「这次改动好了多少，我有几分把握」。

LLM 应用最尴尬的地方是：prompt 改了、模型换了、检索参数调了，然后所有人只能凭印象说"好像好一点"。没有回归测试，没有成本台账，出了问题只能靠翻日志猜。

Lens 是一个自带评估能力的链路追踪工具。一行接入，就能回答三个问题：

1. **这次请求的时间花在哪？** —— 嵌套 span 瀑布图，精确到每一次模型调用
2. **每次调用花了多少钱？** —— 按模型聚合成本，价格表同步自 models.dev
3. **新版真的更好吗？** —— 配对 bootstrap + 置信区间，明确告诉你差异是否显著

## 快速开始

```bash
uv venv && uv pip install -e .
python examples/demo_agent.py     # 生成追踪数据
python examples/run_eval.py       # 跑一次 A/B 评测
python -m lens.server.app         # 打开看板 http://127.0.0.1:8765
```

无需 API key 即可跑通全流程：默认使用确定性 mock provider。

Docker 方式（镜像 263 MB，容器带 HEALTHCHECK）：

```bash
cp .env.example .env             # 按需填 provider 与 key
docker compose up -d             # 打开 http://127.0.0.1:8765
```

数据落在外挂的 `./data` 卷里，`docker compose down` 不会丢 trace。

## 三行接入

```python
from lens import Lens

lens = Lens(project="my-agent")
with lens.trace("answer", input=question) as run:
    docs = retrieve(question)                       # 手动埋点
    reply = lens.llm([{"role": "user", "content": question}])   # 自动记录 tokens/成本/延迟
    run.output = reply.text
```

没包 `trace` 也不会丢数据——直接调 lens.llm() 会被记录成一条单步 run。漏埋点比多埋点代价高得多。

## 一个真实的评测，和它翻车的过程

15 条用例、真实模型（DeepSeek-V3.2）、对比「单次采样」与「best-of-3」。第一次跑出来的结果很漂亮：

| 版本 | 均分 | 95% CI |
|---|---|---|
| v1 单次采样 | 0.200 | [0.033, 0.400] |
| v2 best-of-3 | 0.400 | [0.133, 0.667] |

v2 整整是 v1 的两倍。差一点就写进结论了——直到我点开一条被判 0 分的用例。

那条问「什么时候该从大模型换成小模型」，模型回答得很好：讲清了效率、成本、可控性的权衡。但它得了 0 分，因为我的关键词写的是 `expensive` / `cost`，而模型用中文回答了「成本」。

15 条用例的关键词几乎全是英文。**中文回答被系统性地判成错误答案。**

改掉关键词后，同一批回答、同一个模型，结果完全变了：

| 版本 | 均分 | 95% CI | 通过率 | 成本 | 均延迟 |
|---|---|---|---|---|---|
| v1 单次采样 | 0.933 | [0.800, 1.000] | 93% | $0.0138 | 53 s |
| v2 best-of-3 | 1.000 | [1.000, 1.000] | 100% | $0.0400 | 144 s |

配对比较：差值 **+0.067**，CI `[+0.000, +0.200]`，**p = 0.724 → 无显著差异**，胜负平 1/0/14。

真实结论：**best-of-3 多花 2.9 倍成本、2.7 倍延迟，只补上了 15 条里的一条，统计上测不出差别。**

比这个结论更重要的，是前一个：**评测标准的缺陷比被测对象更能左右结论**。假阴性严重的评测集有两个后果——把噪声当成效果，或者把有效改动判成无效；两种错误都不会自己暴露，只会安静地进入决策。

（v2 的区间塌缩成一个点，是因为 15 条全满分。这本身就是信号：这批用例对 v2 太简单了，没有区分度。要区分两者需要更难的题，而不是更多采样。）

## 设计决策

**span 上下文放在 ContextVar 而不是实例属性。** 并行检索 fan-out 时，每个分支的子 span 必须挂到正确的父节点上。用实例属性存栈，最后结束的协程会覆盖别人的父指针。

**未知模型的价格返回 `None`，不是 0。** 包里同时标记 `attributes.pricing = "unknown"`，看板上显示为 `?`。把算不出来的成本当成 0，会让成本报表系统性偏低，而且没人发现。

**A/B 用配对 bootstrap，不用 t 检验。** 分数有界在 0..1、常呈双峰，正态假设是假的；更重要的是配对能消掉用例难度这个最大方差来源——同一批题上比较，区间会窄得多。

**judge 必须能和人工对齐。** `judge_reliability()` 算 Cohen's Kappa，因为原始一致率会骗人：两个评审都倾向于打"通过"时，90% 的表面一致率背后 kappa 可能只有 0.6。

**看板只读。** 一个能修改自己所报告数据的可观测性工具，没法用来追责。写入只走 SDK，读取只走 HTTP。

**mock provider 是一等公民。** 埋点逻辑、成本计算、统计显著性都应该能在 CI 里零成本验证。真实调用作为可插拔的 provider 存在，而不是测试的前提。

## Provider 配置

所有 provider 都走 OpenAI 兼容协议，切换只改两个变量：

```bash
LENS_PROVIDER=siliconflow   # zenmux / siliconflow / deepseek / moonshot / openai / ollama / mock
LENS_MODEL=deepseek-ai/DeepSeek-V3.2
```

凭证解析顺序：环境变量 → `.env` → 复用本机其他工具已存的 key（自动读取 `opencode/auth.json`）。密钥不会进入版本库，`lens providers` 只显示掩码。

```bash
python -m lens.cli providers   # 查看各 provider 就绪状态与价格表快照
```

代理相关：`LENS_TRUST_ENV=0` 忽略继承的 `HTTP_PROXY`（某些沙箱会注入只供自己使用的回环代理），`LENS_PROXY` 显式指定出口。

## 项目结构

```
lens/
  config.py          provider 解析与凭证来源
  db.py              SQLite schema（WAL，runs/spans/eval_results）
  trace.py           SDK：trace/span/llm，contextvars 上下文
  pricing.py         价格匹配（含跨 provider 回退）
  providers/         openai_compat + mock
  eval/
    judges.py        确定性打分器 + LLM judge（三级解析降级）
    stats.py         bootstrap CI / 配对比较 / kappa / spearman
    runner.py        数据集加载、批量评测、标签对比
  server/            FastAPI（只读）+ 单文件看板
examples/            demo agent、评测演示、示例数据集
```

## 已知限制

- **SQLite 单机**。单机百万级 span 没问题，多实例需要换 Postgres，schema 已按可迁移方式设计。
- **全量记录，无采样**。生产高流量下应加 head-based sampling，目前靠 `MAX_PAYLOAD_CHARS` 截断控制体积。
- **judge 校准缺标注界面**。目前需要在代码里传入人工标签，没有内建的标注工作流。
- **评测是串行等待的**。真实 provider 单次调用实测 25–50 s，15 条用例跑 best-of-3 需要 8 分钟以上。缓解手段（断点续跑、逐条落库、限流感知）已具备，但还没有结果缓存。
- **容器镜像未做多阶段构建**。263 MB 里含完整 Python 运行时，能压到 150 MB 以内，但当前体积不影响使用。

## 数据模型

`runs`（一次执行）→ `spans`（执行内的计时步骤，可嵌套）→ `eval_results`（用例级评分）。

三张表共用一个 SQLite 文件，整个 trace 跟着项目走——不需要先把某个可观测性服务立起来才能开始度量。
