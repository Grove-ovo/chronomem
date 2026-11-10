# ChronoBench-100 四基线横向评测报告（ChronoMem）

- 生成时间：2026-11-10 17:01 UTC
- 数据集：100 组多会话冲突演化样本（reversal×20、partial_update×20、retraction×20、as_of×20、anchor_drift×20），共 160 条评测查询
- 运行环境：Python 3.10.11 / SQLite 3.40.1
- 随机种子：20261111（数据集与基线全部确定性可复现）

### 评测方法（确定性 harness，隔离记忆管理能力）

四套系统观察**同一条多会话对话流**（每轮携带结构化 statement 元数据，
对全部系统同等可见，从而隔离记忆管理能力、排除 NLU 抽取差异干扰）：

| 系统 | 记忆组织方式 | 时间感知 | 代码感知 |
|---|---|---|---|
| **ChronoMem** | MVCC 版本链 + 四态冲突消解 + 锚点指纹 | `valid_from/to_turn` 精确窗口 | AST 符号 SHA256 指纹 |
| Full-Context | 全量原始对话扫描 | 按轮次过滤 | 无 |
| Rolling Summary | 会话摘要窗口（最近 3 会话） | 无（只有最新态） | 无 |
| Naive Vector RAG | 全量切块 TF-IDF 向量 Top-K | 无 | 无 |

指标定义严格遵循 PRD §2：

* **Temporal_State_Acc** = 严格采纳最新有效状态作答的查询数 / 经历过状态覆写的查询总数（reversal 类）；
* **Stale_Poisoning_Rate** = 当前态查询注入上下文的过期/撤销/漂移条目数 / 注入上下文总条数（逐条按 ground-truth 演化账本审计）；
* **Slot_Merge_F1** = 局部修改后最终槽位集合 vs Ground-Truth 的字段级 P/R/F1（partial_update 类，值错误的字段同时计入 FP 与 FN）；
* **Abstention_Acc** = 拒答类查询（已撤销 / 从未提及 / 锚点漂移）中准确识别并拒答的比例（严格模式：拒答原因也必须正确）。


## 1. 四大核心指标总表

| 系统 | Temporal_State_Acc ↑ | Stale_Poisoning_Rate ↓ | Slot_Merge_F1 ↑ | Abstention_Acc ↑ | As_Of_Acc ↑ | 误拒答率 ↓ | 平均上下文 Tokens/查询 ↓ |
|---|---|---|---|---|---|---|---|
| **ChronoMem** | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% | 13 |
| **Full-Context** | 100.0% | 58.5% | 100.0% | 66.7% | 100.0% | 0.0% | 47 |
| **Rolling Summary** | 100.0% | 56.4% | 91.4% | 66.7% | 0.0% | 0.0% | 43 |
| **Naive Vector RAG** | 10.0% | 58.6% | 36.3% | 0.0% | 50.0% | 0.0% | 76 |

**关键结论**：ChronoMem 将当前态查询的过期记忆污染率压降至 **0.0%**（Naive Vector RAG 为 58.6%，Full-Context 为 58.5%），同时在最新状态准确率、局部合并 F1、时间旅行问答与拒答率上全面领先 Naive Vector RAG。

## 2. 分类别表现

### Cat-1 单属性多次反转（3~5 次改口）

| 系统 | Accuracy（当前态严格最新） |
|---|---|
| **ChronoMem** | 100.0% |
| **Full-Context** | 100.0% |
| **Rolling Summary** | 100.0% |
| **Naive Vector RAG** | 10.0% |

### Cat-2 多字段配置局部增量修改

| 系统 | Slot_Merge_F1 |
|---|---|
| **ChronoMem** | 100.0% |
| **Full-Context** | 100.0% |
| **Rolling Summary** | 91.4% |
| **Naive Vector RAG** | 36.3% |

### Cat-3 显式撤销与无证据拒答陷阱

| 系统 | Accuracy（拒答原因严格匹配） |
|---|---|
| **ChronoMem** | 100.0% |
| **Full-Context** | 100.0% |
| **Rolling Summary** | 100.0% |
| **Naive Vector RAG** | 0.0% |

### Cat-4 历史时点回溯（时间旅行）

| 系统 | As_Of_Acc |
|---|---|
| **ChronoMem** | 100.0% |
| **Full-Context** | 100.0% |
| **Rolling Summary** | 0.0% |
| **Naive Vector RAG** | 50.0% |

### Cat-5 代码重构锚点漂移

| 系统 | Accuracy（漂移拒答+控制组正常作答联合） |
|---|---|
| **ChronoMem** | 100.0% |
| **Full-Context** | 50.0% |
| **Rolling Summary** | 50.0% |
| **Naive Vector RAG** | 50.0% |

## 3. 过期记忆污染审计明细

| 系统 | 注入上下文条目总数 | 其中过期/撤销/漂移条目 | 污染率 |
|---|---|---|---|
| **ChronoMem** | 120 | 0 | 0.0% |
| **Full-Context** | 236 | 138 | 58.5% |
| **Rolling Summary** | 225 | 127 | 56.4% |
| **Naive Vector RAG** | 280 | 164 | 58.6% |

> ChronoMem 的当前态检索硬性过滤 `status != 'active'` 的全部版本，`retracted` / `anchor_drifted` 槽位以显式状态提示（notice）返回而非检索结果，因此污染率结构性为 0。

## 4. 数据集构成

| 类别 | 样本数 | 考察能力 |
|---|---|---|
| reversal | 20 | 跨会话偏好反转后只采纳最新 active 状态 |
| partial_update | 20 | 局部字段修改时未提及字段零丢失（MERGE_PARTIAL 深度合并） |
| retraction | 20 | 显式撤销/从未提及的槽位正确拒答 |
| as_of | 20 | valid_from/to 窗口精确还原历史世界状态 |
| anchor_drift | 20 | 绑定文件/AST 符号被重构后拒绝引用失效记忆 |

## 5. 复现

```bash
chronomem bench                 # 全量 100 组评测，生成 reports/chronobench_results.md
chronomem visualize --demo      # 生成记忆版本链时间轴可视化 visualizer/index.html
pytest tests/ -q                # 74 项单元测试（含 MVCC 不变量随机风暴）
```
