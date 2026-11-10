---
kind: spec
title: "PRD 2 (Markdown): ChronoMem — 抗过期污染的时效状态与代码锚点 AI 长期记忆引擎"
comments: none
---

# 产品需求文档 (PRD)：ChronoMem

基于 MVCC 多版本状态链与代码指纹锚点的抗过期污染 AI 长期记忆引擎（附 `ChronoBench-100` 跨会话冲突评测基准）。
配套可视化架构图与交互式设计见 [ChronoMem 架构与评测蓝图](../prd-chronomem)。

---

## 1. 项目定位与目标 (Goal & Positioning)

### 1.1 解决的核心问题
现有开源 AI 记忆框架（如早期 Mem0、朴素向量 RAG、滚动摘要）将历史对话切块存入向量库并按语义相似度召回，在跨越数十轮会话的真实工程与助理场景中存在三大失真：
1. **新旧状态并存污染 (Temporal State Poisoning)**：当用户在后续会话中推翻或修改早期设定（如技术栈迁移、退款规则变更、部署参数调整）时，向量检索同时召回新旧两条高相似度记录，导致 Agent 依据过期事实做出错误决策。
2. **局部修改丢失细节 (Partial Update Loss)**：当用户对包含多个字段的复杂状态进行局部修改（如“仅将超时时间改为 30s，其余重试配置不变”）时，扁平文本记忆要么丢失未提及的原字段，要么分裂为矛盾碎片。
3. **外部代码/文档漂移导致记忆失效 (External Anchor Drift)**：当记忆关联的底层代码文件或函数签名已在 Git 仓库中被重构或删除时，静态记忆库毫无察觉，继续向 Agent 注入已失效的过期路径与接口信息。

### 1.2 最终交付目标 (Definition of Done)
构建一个零重型外部数据库依赖（基于 Python + SQLite + FTS5 + 轻量混合检索）的独立长期记忆引擎 `chronomem` 与跨会话冲突评测集 `ChronoBench-100`，实现：
- **MVCC 多版本原子状态存储与四态冲突消解器**（`ADD` / `SUPERSEDE` / `MERGE_PARTIAL` / `RETRACT`）；
- **文件与 AST 符号指纹锚点联动失效检测器 (`Anchor Watcher`)**；
- **时间旅行查询 (`as_of_turn`) 与当前态零污染检索**；
- 使用真实 Token 在 100 组多会话冲突演化测试流上完成 **四大基线横向评测**（`Full-Context` vs `Naive Vector RAG` vs `Rolling Summary` vs `ChronoMem`），生成完整实测报告与记忆版本演化可视化器。

---

## 2. 核心评测指标与确定性计算公式

1. **最新状态准确率 (Temporal_State_Acc)**
   - 定义：`严格采纳最新有效状态作答的查询数 / 经历过状态覆写的查询总数`
   - 意义：衡量经历 2 至 5 次跨会话偏好/配置反转后，系统是否只使用最新 `active` 状态。

2. **过期记忆污染率 (Stale_Poisoning_Rate)**
   - 定义：`当前态查询 Top-K 召回结果中包含 stale/retracted/anchor_drifted 记录的条数 / Top-K 召回总条数`
   - 意义：直接量化检索层是否将已作废或代码已失效的旧记忆泄露进 Agent 当前上下文。

3. **局部修改槽位保真度 (Slot_Merge_F1)**
   - 定义：针对多字段结构化状态的局部更新任务，计算最终状态槽位集合与 Ground-Truth 槽位集合的精确率 `P`、召回率 `R` 及 `F1 = 2 * P * R / (P + R)`。
   - 意义：验证局部修改时“已改字段精准更新、未改字段零丢失”。

4. **无证据与已撤销拒答率 (Abstention_Acc)**
   - 定义：`准确识别“该设定已撤销/从未提及/关联代码已失效”并拒答的查询数 / 拒答类查询总数`
   - 意义：对应 LongMemEval 中最难的 Abstention 维度，防止记忆层诱发幻觉。

---

## 3. 功能模块规格 (Functional Specifications)

### 3.1 模块 M1：MVCC 原子状态存储引擎 (`chronomem/store.py`)
使用 SQLite 实现支持多版本并发控制（MVCC 思想）的记忆存储表 `memory_atoms`：
- 核心字段：
  - `atom_id` (主键)、`namespace`（会话/项目隔离域）、`entity`（实体标识）、`slot`（属性槽位）、`value_json`（结构化值或原子断言文本）、`summary_text`（供 FTS5 检索的归一化文本）；
  - 生命周期字段：`status`（枚举：`active` | `stale` | `retracted` | `anchor_drifted`）、`valid_from_turn`（生效轮次）、`valid_to_turn`（失效轮次，`active` 时为 NULL）、`superseded_by`（指向替代它的新 `atom_id`）、`supersedes`（指向被其替代的旧 `atom_id`）；
  - 溯源与锚点字段：`source_turn_id`（原始对话轮次 ID）、`anchor_path`（可选，绑定的文件路径或 `file.py::symbol`）、`anchor_sha256`（写入时的文件或 AST 节点 SHA256 指纹）。
- 核心数据库不变量（Invariant）：对于任意 `(namespace, entity, slot)`，在任一时刻最多只允许存在 **1 条** `status == 'active'` 的记录。

### 3.2 模块 M2：写入期四态冲突消解状态机 (`chronomem/resolver.py`)
当新对话轮次 `Turn_t` 进入系统时，提取候选状态变更并对比当前 `active` 记录，执行以下 4 种确定性状态转移之一：
1. **ADD（新增）**：当前 `(entity, slot)` 无 `active` 记录，直接插入新记录，`status = 'active'`，`valid_from_turn = t`。
2. **SUPERSEDE（覆盖替代）**：当前 `(entity, slot)` 已有 `active` 记录且新值为互斥覆写。在单个事务内将旧记录更新为 `status = 'stale'`、`valid_to_turn = t`、`superseded_by = new_id`，同时插入新 `active` 记录并设置 `supersedes = old_id`。
3. **MERGE_PARTIAL（局部槽位合并）**：当目标槽位为复合字典/列表且新输入仅修改子属性时，读取旧 `active` 记录的 `value_json`，执行深度增量合并生成完整新快照，然后走 `SUPERSEDE` 版本链更新，确保历史版本可回溯且当前版本字段完整。
4. **RETRACT（显式撤销）**：当用户明确取消或作废某项设定时，将原 `active` 记录更新为 `status = 'retracted'`、`valid_to_turn = t`，并在检索时标记该槽位曾被显式撤销。

### 3.3 模块 M3：代码与文件指纹锚点联动器 (`chronomem/anchor_watcher.py`)
- 支持在写入工程记忆时绑定文件路径或 Python AST 节点锚点（例如 `src/payment.py::RefundService.process`），自动计算并持久化该 AST 节点或文件的 SHA256 哈希。
- 在执行记忆检索前（或调用 `chronomem verify-anchors` 时），自动扫描绑定的本地工作区文件：
  - 若目标文件被删除、目标 AST 符号消失、或节点内容哈希发生变化，自动将该条记忆状态更新为 `anchor_drifted`，避免 Agent 盲目引用过期代码结构。

### 3.4 模块 M4：时效感知检索器与时间旅行接口 (`chronomem/retriever.py`)
- **当前态检索 (`retrieve_current`)**：结合 `(entity, slot)` 精确匹配、FTS5 全文检索（支持中文二元分词与英文词干）及语义打分，硬性过滤掉所有 `status != 'active'` 的过期记录；对 `anchor_drifted` 或 `retracted` 槽位返回显式状态提示，辅助 Agent 正确拒答或重新核实。
- **历史时间旅行检索 (`retrieve_as_of(turn_id)`)**：利用 `valid_from_turn <= turn_id AND (valid_to_turn IS NULL OR valid_to_turn > turn_id)` 精确还原任意历史轮次 `turn_id` 时的世界状态快照，精准回答“第 N 轮改动之前我们使用的是什么配置”类时间推理问题。

### 3.5 模块 M5：ChronoBench-100 数据集与四基线评测流水线 (`benchmarks/` & `chronomem/cli.py`)
- 构建 100 组多会话冲突演化测试样本（涵盖：5 大场景 × 20 组 = 100 组）：
  1. 单属性多次反转（3~5 次改口）；
  2. 多字段复杂配置局部增量修改；
  3. 显式撤销与无证据拒答陷阱；
  4. 历史时点回溯提问（`as_of` 时间旅行推理）；
  5. 代码重构引发关联记忆锚点漂移失效。
- 实现 3 个对照基线与 `ChronoMem` 同台竞技：
  - `Baseline 1`: Naive Vector / Chunk RAG（全量切块相似度 Top-K 召回，不过滤历史版本）；
  - `Baseline 2`: Rolling Summary Memory（滚动文本摘要覆盖）；
  - `Baseline 3`: Full-Context Window（全量对话硬塞）；
  - `Ours`: `ChronoMem`（MVCC 状态链 + 局部合并 + 代码锚点校验）。

---

## 4. Git 管理、Issue 驱动、PR 与合并规范 (Mandatory Git & GitHub Workflow)

本项目全生命周期必须严格执行工业级 Git 与 GitHub 研发流：

1. **仓库初始化与分支纪律**
   - 初始化独立 Git 仓库，主分支为 `main`，严禁在 `main` 分支直接提交未验证代码。
   - 优先使用 `gh` CLI 连接远端 GitHub 仓库执行标准 Issue 与 PR 操作；若无远端访问权限，则在本地维护结构化 Issue 追踪记录并通过特性分支 + `--no-ff` Merge Commit 严格执行分支隔离与合并门禁。

2. **里程碑与 Issue 拆解（开工前必做）**
   - 动手编码前，先依据本 PRD 创建以下 6 个核心 Issues：
     - `Issue #1`: 项目脚手架、MVCC SQLite 存储引擎 (`store.py`) 与数据库不变量单测
     - `Issue #2`: 实现写入期四态冲突消解器 (`ADD / SUPERSEDE / MERGE_PARTIAL / RETRACT`)
     - `Issue #3`: 实现代码与 AST 符号指纹锚点联动失效检测器 (`anchor_watcher.py`)
     - `Issue #4`: 实现时效感知混合检索器与 `as_of` 时间旅行查询接口 (`retriever.py`)
     - `Issue #5`: 构建 `ChronoBench-100` 跨会话冲突评测集与三大对照基线实现
     - `Issue #6`: 运行完整四基线实测实验、生成版本链时间轴可视化器 (`visualizer/`) 与完整 README

3. **特性分支开发、拉取 PR 与合并闭环**
   - 每个 `Issue #N` 必须遵循：
     1. 从 `main` 切出分支 `git checkout -b feat/issue-N-<slug>`；
     2. 编写实现代码与单元测试，确保 `pytest` 100% 通过；
     3. 提交清晰的 Conventional Commit（如 `feat(resolver): implement 4-state MVCC conflict resolution (#2)`）；
     4. 创建 Pull Request（`gh pr create`），在 PR 描述中贴出单元测试与指标验证结果；
     5. 校验通过后合并 PR 回 `main` 并删除特性分支，关闭对应 Issue，再进入下一个 Issue。

---

## 5. P0 最终完成验收标准 (Final Acceptance Gate)

Agent 在 Goal 模式下必须持续推进，直到以下全部条件同时满足方可宣告完成：
1. **核心单测与 MVCC 不变量 100% 通过**：`pytest` 全绿，验证任意并发或连续覆写后同一 `(entity, slot)` 绝不出现两条 `active` 记录。
2. **ChronoBench-100 完整跑分完成**：100 组多会话冲突演化测试集全部跑完，生成真实的四基线对比数据表（`reports/chronobench_results.md`），实测证明 `ChronoMem` 将过期记忆污染率压降至接近 `0%`，并在最新状态准确率、局部合并 F1、时间旅行问答与代码漂移拒答率上全面显著领先 `Naive Vector RAG`。
3. **可视化器与 CLI 可直接运行**：命令行 `chronomem bench`、`chronomem inspect` 与静态版本演化可视化网页全部验证可用。
4. **Git 历史与 PR 闭环完整**：全部 6 个 Issues 均按 `Issue → Branch → PR → Merge` 闭环完成，`git status` 干净。
