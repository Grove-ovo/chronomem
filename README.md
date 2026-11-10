# ChronoMem

**抗过期污染的时效状态与代码锚点 AI 长期记忆引擎** —— 基于 MVCC 多版本状态链、写入期四态冲突消解与代码指纹锚点的独立长期记忆库，附跨会话冲突评测基准 `ChronoBench-100`。

> 现有开源 AI 记忆框架（早期 Mem0、朴素向量 RAG、滚动摘要）在数十轮跨会话场景下存在三大失真：
> 1. **新旧状态并存污染**（Temporal State Poisoning）—— 向量检索同时召回新旧两条高相似度记录，Agent 依据过期事实做出错误决策；
> 2. **局部修改丢失细节**（Partial Update Loss）—— "只改超时时间、其余不变"后，扁平文本记忆要么丢字段、要么分裂成矛盾碎片；
> 3. **外部代码漂移记忆失效**（External Anchor Drift）—— 记忆绑定的函数签名已被重构/删除，记忆库毫无察觉地继续注入失效路径。
>
> ChronoMem 用**数据库级不变量**逐一根治这三类问题。

---

## 核心能力

| 模块 | 实现 | 关键性质 |
|---|---|---|
| **M1** MVCC 存储引擎 (`chronomem/store.py`) | SQLite `memory_atoms` 多版本表 | 任意操作序列后，同一 `(namespace, entity, slot)` **至多 1 条 `active` 记录**（partial unique index 数据库级强制）；`supersedes`/`superseded_by` 双向链完整可回溯 |
| **M2** 四态冲突消解器 (`chronomem/resolver.py`) | `ADD` / `SUPERSEDE` / `MERGE_PARTIAL` / `RETRACT` | 单事务原子版本链转移；字典子集输入自动深度增量合并，未提及字段零丢失 |
| **M3** 锚点联动失效检测 (`chronomem/anchor_watcher.py`) | 文件 / Python AST 符号 SHA256 指纹 | 文件删除、符号消失、内容哈希变化 → 自动转 `anchor_drifted`，检索前自动扫描 |
| **M4** 时效感知检索器 (`chronomem/retriever.py`) | FTS5（中文二元分词 + 英文轻量词干）BM25 ⊕ `(entity, slot)` 精确匹配 ⊕ token 重叠 | **硬性过滤 `status != 'active'`**（结构性零污染）；`retracted`/`anchor_drifted` 以显式 notice 返回辅助拒答；`retrieve_as_of(turn)` 按 MVCC 可见性窗口精确还原任意历史轮次 |
| **M5** ChronoBench-100 + CLI (`benchmarks/` & `chronomem/cli.py`) | 100 组多会话冲突演化样本 × 5 大类；三大对照基线 | 确定性可复现（固定种子）；污染逐条 ground-truth 审计 |

## ChronoBench-100 实测结果

四套系统观察同一条多会话对话流（结构化 statement 元数据对全部系统同等可见，
隔离记忆管理能力、排除 NLU 差异），160 条评测查询：

| 系统 | Temporal_State_Acc ↑ | Stale_Poisoning_Rate ↓ | Slot_Merge_F1 ↑ | Abstention_Acc ↑ | As_Of_Acc ↑ | 平均上下文 Tokens/查询 |
|---|---|---|---|---|---|---|
| **ChronoMem** | **100.0%** | **0.0%** | **100.0%** | **100.0%** | **100.0%** | **13** |
| Full-Context | 100.0% | 58.5% | 100.0% | 66.7% | 100.0% | 47 |
| Rolling Summary | 100.0% | 56.4% | 91.4% | 66.7% | 0.0% | 43 |
| Naive Vector RAG | 10.0% | 58.6% | 36.3% | 0.0% | 50.0% | 76 |

* ChronoMem 将过期记忆污染率压降至 **0.0%**（结构性的：当前态检索只可能返回 `active` 记录）；
* Naive Vector RAG 在 3~5 次跨会话改口后最新状态准确率仅 **10%**（新旧切块相似度打平，召回被旧版本支配）；
* Rolling Summary 无时间轴（时间旅行问答 **0%**），摘要窗口丢弃早期配置字段（Slot_Merge_F1 降至 91.4%）；
* Full-Context 精度打平但污染 58.5%，且**无法感知代码漂移**（拒答率仅 66.7%）。

完整报告：[reports/chronobench_results.md](reports/chronobench_results.md)（含分项方法说明、分类别表现、污染审计明细）。

### 数据集构成（5 大类 × 20 组）

| 类别 | 考察能力 |
|---|---|
| `reversal` | 单属性跨会话 3~5 次反转后只采纳最新 `active` 状态 |
| `partial_update` | 多字段配置局部增量修改，未提及字段零丢失（`MERGE_PARTIAL`） |
| `retraction` | 显式撤销 / 从未提及槽位的正确拒答 |
| `as_of` | `valid_from_turn ≤ t < valid_to_turn` 窗口精确时间旅行 |
| `anchor_drift` | 重构（重命名符号 / 删除方法 / 删除文件）后拒绝引用失效记忆，且不误伤存活锚点 |

## 快速开始

```bash
# 无第三方运行时依赖（Python ≥ 3.10 标准库即可），pytest 用于测试
pip install -e .

# 写入与四态消解
chronomem add --entity user --slot db --value '"MySQL"' --turn 1
chronomem add --entity user --slot db --value '"PostgreSQL"' --turn 4   # SUPERSEDE
chronomem add --entity svc --slot retry_config \
  --value '{"timeout_s": 30}' --turn 5                                  # MERGE_PARTIAL（其余字段自动保留）
chronomem retract --entity user --slot plan --turn 6                    # RETRACT

# 当前态零污染检索（retracted/anchor_drifted 槽位给出显式 notice）
chronomem retrieve "user 的数据库是什么" --entity user --slot db

# 时间旅行：还原第 3 轮时的世界状态
chronomem as-of 3 "user db"

# 代码锚点绑定与漂移检测
chronomem add --entity svc --slot refund_entry --value '{"entry": "RefundService.process"}' \
  --anchor "src/payment.py::RefundService.process" --workspace .
chronomem verify-anchors .

# 检视版本链
chronomem inspect
```

`inspect` 输出示例：

```
[default] user · db
  #1    stale          t1→4    "MySQL"          superseded_by #5
  #5    PostgreSQL     active  t4→∞             supersedes #1
```

## 可视化器

```bash
chronomem visualize --demo            # 内置演示数据 → visualizer/index.html
chronomem visualize --db my.db        # 渲染你自己的记忆库
```

生成**自包含**的 `visualizer/index.html`（零外部资源，离线可用）：每条
`(namespace, entity, slot)` 版本链渲染为一条时间轴 lane，版本段按生命周期着色
（`active` 绿 / `stale` 灰 / `retracted` 红 / `anchor_drifted` 橙），支持
namespace 过滤、entity/值搜索与悬停详情（value JSON、锚点路径、supersedes 链）。

## 项目结构

```
chronomem/
  store.py           # M1: MVCC 存储引擎 + 单一 active 不变量（partial unique index）
  resolver.py        # M2: 四态冲突消解状态机
  anchor_watcher.py  # M3: 文件/AST 符号指纹锚点联动失效检测
  retriever.py       # M4: 时效感知混合检索 + as_of 时间旅行
  tokenizer.py       # 中文二元分词 + 英文轻量词干（双端一致）
  visualizer.py      # 版本链时间轴 HTML 生成器
  cli.py             # add / retract / retrieve / as-of / verify-anchors / inspect / bench / visualize
benchmarks/
  dataset.py         # ChronoBench-100 确定性生成器（5 大类 × 20 组）
  systems.py         # ChronoMem + 3 基线（Naive Vector RAG / Rolling Summary / Full-Context）
  metrics.py         # 四大指标确定性计算
  runner.py          # 评测编排 + ground-truth 演化账本（污染审计）
  report.py          # reports/chronobench_results.md 生成
  data/chronobench_100.json
reports/chronobench_results.md
visualizer/index.html
tests/               # 74 项单测（含 500 步随机操作风暴 + 6 线程并发不变量校验）
```

## 测试与复现

```bash
pytest tests/ -q                 # 74 passed
chronomem bench                  # 全量 100 组四基线评测 → reports/
chronomem bench --samples 10     # 快速子集冒烟
```

MVCC 不变量在任何操作序列下成立，由三层共同保证：

1. **数据库层**：`CREATE UNIQUE INDEX uq_one_active_per_key ON memory_atoms(namespace, entity, slot) WHERE status='active'`；
2. **事务层**：`MemoryStore.supersede()` 在单事务内完成 关闭旧窗口 → 插入新 active → 回填 `superseded_by`；
3. **校验层**：`verify_invariants()` 检查单一 active、窗口闭合、链双向一致，单测在每步随机操作后断言。

## 研发流程（Issue → Branch → PR → Merge）

本项目严格遵循 Issue-Driven Development，全部功能经 GitHub PR 合入 `main`，
见 [Issues](https://github.com/Grove-ovo/chronomem/issues) 与
[Pull Requests](https://github.com/Grove-ovo/chronomem/pulls)：

| Issue | 内容 | PR |
|---|---|---|
| #1 | MVCC 存储引擎 + 不变量单测 | [#7](https://github.com/Grove-ovo/chronomem/pull/7) |
| #2 | 四态冲突消解状态机 | [#8](https://github.com/Grove-ovo/chronomem/pull/8) |
| #3 | AST 锚点失效检测器 | [#9](https://github.com/Grove-ovo/chronomem/pull/9) |
| #4 | 时效感知检索器 + as_of | [#10](https://github.com/Grove-ovo/chronomem/pull/10) |
| #5 | ChronoBench-100 + 三大基线 | [#11](https://github.com/Grove-ovo/chronomem/pull/11) |
| #6 | 四基线实测报告 + 可视化器 + README | [#12](https://github.com/Grove-ovo/chronomem/pull/12) |

## License

MIT
