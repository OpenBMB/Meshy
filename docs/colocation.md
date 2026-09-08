# Colocation 环形调度共置 GPU

## 概述

**Colocation** 让多个 GPU Service（Inference / Training）分时复用同一批 GPU。任一时刻只有一个 Service 持有 GPU，所有权以**令牌**的形式在预先配置好的**环**中传递。

与传统做法（由某个中心角色调用其他角色的 HTTP 端点来抢占显存）不同，本实现是**去中心化**的：

- 调度状态是 TransferQueue 上的一张**请求账本**（每个请求一行，持有者原地写入 grant），每个 ServiceGroup leader 独立扫描并还原出**相同**的状态；
- 只有当前持有者有权发布下一次转移，因此不需要 Controller 参与每一步；
- `ColocationManager` 只管调度，**不执行任何显存操作**——显存的获取与释放由各 Service 以回调形式注入。

### 与其他模式对比

| 维度 | Colocation（本文） | Disaggregate |
| --- | --- | --- |
| **GPU 布局** | 环成员分时复用同一批卡 | 各 Service 独占各自的卡 |
| **调度决策** | 各 manager 本地扫描账本 | 无需调度 |
| **参与角色数** | 任意（2 或更多） | — |
| **可审计性** | grant `sequence` 严格连续，可去重、可重放 | — |

### 何时使用

选择 **Colocation**：GPU 资源紧张，可以接受生成与训练串行执行。

选择 **Disaggregate**：GPU 资源充足，希望生成与训练真正并行。此时不要配置 `COLOCATIONS`；所有 Service 会获得 `NoopColocationManager`，调度逻辑整体旁路。

---

## 快速开始

在 recipe 中声明一个环，并把它传给 `Ignitor`：

```python
from meshy.service.colocation import ColocationRing, SchedulingMode

COLOCATIONS = [
    ColocationRing(
        group_id="actor_card",
        ring=(
            ("actor_infer", SchedulingMode.FALLBACK),
            ("actor_train", SchedulingMode.ON_DEMAND),
        ),
    )
]


def main() -> None:
    Ignitor(SERVICE_GROUPS, COLOCATIONS).run()
```

`ring` 中的 id 是 **ServiceGroup id**，不是副本名；整个 ServiceGroup 作为一个整体参与调度。加入第三个角色只需在环中再加一项，无需改动任何 Service 代码。

共卡的 `ServiceGroup` **必须**属于某个环：`TitanTrainingService.from_info` 会通过 `require_ring` 校验，未声明环直接报错（旧的 trainer 驱动 HTTP 仲裁路径已删除）。

---

## 配置

### `ColocationRing`

| 参数 | 用途 | 默认值 |
| --- | --- | --- |
| `group_id` | 环标识，用于隔离该环的 TQ 账本分区 | 必填 |
| `ring` | 有序的 `(service_id, mode)` 成员列表，至少两项，id 唯一 | 必填 |
| `poll_interval` | manager 线程扫描账本的间隔（秒） | `1.0` |

`ring` 的每一项会被归一化为 `RingNode(service_id, mode)`。在 recipe 中写普通二元组即可，读取时以下两种写法都成立：

```python
config.ring[0].service_id     # "actor_infer"
service_id, mode = config.ring[0]
```

`config.initial_owner` 是 `ring[0]`，genesis 会把 GPU 交给它。

### 调度模式

`SchedulingMode` 描述的是**获得 GPU 的资格条件**。

| 模式 | 使用者 | 语义 |
| --- | --- | --- |
| `ON_DEMAND` | Training | 必须先发布 `GpuRequest`（priority 0）才可能被选为下一个持有者；任务完成后主动释放 |
| `FALLBACK` | Inference | manager 自动为它维持一条常驻请求（priority 1），没有任何 `ON_DEMAND` 请求时即可获得 GPU。持有期间一旦出现显式请求，会在安全边界让出 |

::: tip
环中至少要有一个 `FALLBACK` 成员。若全部是 `ON_DEMAND`，请求耗尽后没有人会再拿到 GPU。
:::

### 拓扑约束

`build_topology`（`meshy/service/topology.py:_attach_colocations`）在启动前校验：

- 环的 `group_id` 不重复；
- 环成员必须是已声明的 ServiceGroup；
- 同一个 ServiceGroup 只能属于一个环；
- **环成员必须使用完全相同的一组卡**——否则令牌无法表达互斥语义；
- 环成员必须处于 colocate 布局（`is_colocate`）。

任一条不满足都会在启动阶段直接报错，而非运行时才暴露。通过校验后，每个成员组的 `replica_idx == 0` 副本被标记为 `is_colocation_leader`。

---

## 工作原理

### 组成

每个角色都是一个 Service，并持有自己的 `ColocationManager`：

```text
Service
├── Engine            底层计算逻辑，提供 on_colocate_acquire / on_colocate_release
├── Worker            TQ 消费与业务流程，调用 request_gpu / wait_for_grant / release
└── ColocationManager 令牌状态机（显存操作以回调注入）
```

### 控制消息

两种消息落在同一张账本分区 `control.colocate.<group_id>.request`（`meshy/transferqueue/colocation.py:TQRequestLedgerTransport`）。一个请求是一行；当前持有者授权时**原地更新**这一行的 grant 列，因此扫描时用 `force_fetch` 重读全表，并过滤掉预分配但尚未生产的槽位。

```python
@dataclass(frozen=True)
class GpuRequest:
    group_id: str
    service_id: str
    request_id: str
    created_at_ns: int
    payload_ref: str | None = None
    ring_index: int = 0            # 请求方在环中的位置
    priority: int = 0              # 0 = ON_DEMAND，1 = FALLBACK
    kind: RequestKind = RequestKind.ON_DEMAND


@dataclass(frozen=True)
class GpuGrant:
    group_id: str
    sequence: int
    source: str                    # 让出方
    target: str                    # 接手方
    request_id: str | None = None
    transition: str = ""           # 人类可读的转移原因
    payload_ref: str | None = None # 训练 → 推理时携带新 ckpt 路径
```

`GpuGrant` 表示所有权**已经**从 `source` 转移给 `target`。接手方收到后先在本地执行 `on_acquire`（恢复显存），恢复完成才唤醒等待该请求的 Worker——不再额外发布 Ready 消息。

**`payload_ref` 是共卡权重同步的通道**：`TitanWorker` 在 `release` 时把刚落盘的 HF ckpt 路径挂在 grant 上，`SGLangEngine.on_colocate_acquire` 从中读取路径并 `update_weights_from_disk`。只有 genesis（`sequence == 0`）允许没有 `payload_ref`（回落到初始 `model_path`），其余 grant 缺少路径直接报错，防止静默用旧权重生成。

### 选择算法

只有当前持有者做选择。候选是账本里状态为 `open`、尚未授权、且不属于自己的请求，按下面的键排序取第一个：

```python
key = (
    request.priority,                                 # 1. ON_DEMAND(0) 优先于 FALLBACK(1)
    (request.ring_index - owner_index) % ring_size,   # 2. 从环中下一个节点开始的距离
    request.created_at_ns,                            # 3. 先来先得
    request.request_id,
)
```

没有候选时：若上一次 grant 的 `source` 不是自己，为它合成一条 fallback 归还请求再选一次；仍无候选则原地保持所有权，不会产生无意义的令牌循环。

`FALLBACK` 持有者的让出是**主动抢占**：manager 每轮扫描时若发现任何 `open` 请求且自己是 FALLBACK 且持卡，就立即 `_transfer(transition="preempt")`。

### 请求去重

已授权请求的 `sequence` 记录在 `_acquired_sequences`，manager 对已见过的 grant 直接跳过；重复投递同一 `request_id` 不会产生第二次授权。

### 转移协议

```text
1. 当前持有者停止接收新任务
2. 在安全边界停止计算
3. 保存必要状态
4. 释放 GPU 显存（on_release）  ← 必须在第 6 步之前完成
5. 关闭自己的请求行（close_request）
6. 发布 GpuGrant（原地写入目标请求行）
7. 接手方扫描到 GpuGrant
8. 接手方恢复 Engine 与显存（on_acquire）
9. 接手方唤醒本地 Worker 开始计算
```

::: warning
禁止在释放显存前发布 `GpuGrant`。令牌保证的是**计算互斥**，并不保证所有中间缓冲区绝对无重叠。
:::

### 显存策略

- **让出方无条件释放自己占用的全部显存**，不判断谁来接手。`on_release(target)` 的 `target` 只是下一位持有者的身份，不能用来决定只释放哪一部分。
- **接手方自行恢复到可计算状态**，不依赖前一任替它准备。恢复动作按角色不同：Titan 从 CPU `restore_to_gpu`；SGLang 需要 `resume(weights)` → 从 `payload_ref` 加载 ckpt → `resume(kv_cache)` → `continue_generation`。

无条件释放使每条边的正确性**与环的长度无关**；向环中新增角色不需要改动任何既有代码。

### 启动与 Genesis

```text
Ignitor 按声明顺序点火本卡 Service（推理先就绪并释放显存，训练建模后 offload）
↓
dist.barrier()（确保所有 manager 都已就绪）
↓
Ignitor rank 0 发布 genesis Grant（sequence=0，source="authority"，target=ring[0]）
↓
初始持有者恢复 GPU 并唤醒本地 Worker
↓
进入正常调度（此后不再需要 Ignitor 参与）
```

此后的每次转移都由当时的持有者完成，不保留独立的 Authority 对象。

---

## 新 Service 接入指南

接入 Colocation 时，可以把代码分成三部分：

- **框架代码**：仓库已经实现了 Manager、TQ 传输和调度状态机；
- **Service 适配代码**：新 Service 提供显存恢复、释放等回调，并管理 Manager 的生命周期；
- **业务调用代码**：需要 GPU 时申请，获得授权后执行任务，完成后释放 GPU。

新 Service 不需要直接操作 TQ，也不需要重新实现 token 状态机。

### 框架代码：Manager 工厂和调度接口

Service 基类已经提供工厂方法：

```python
# meshy/service/base.py -> Service.build_colocation_manager()
self.colocation = self.build_colocation_manager(
    on_acquire=on_acquire,
    on_release=on_release,
)
self.colocation.start()   # 调用方负责启动
```

`build_colocation_manager()` 会根据拓扑返回合适的 Manager：

- 环内且是 ServiceGroup leader：真正的 `ColocationManager`，传输层为 `TQRequestLedgerTransport`；
- 环外或非 leader：`NoopColocationManager`。

以下由仓库内部完成，Service 不需要直接调用：

```python
# meshy/service/colocation.py        -> ColocationManager
# meshy/transferqueue/colocation.py  -> TQRequestLedgerTransport
transport.create_request(...)
transport.scan_requests()
transport.grant_request(...)
transport.close_request(...)
```

`ColocationManager` 提供的业务接口：

| 方法 | 由谁调用 | 作用 |
| --- | --- | --- |
| `start()` | Service 启动阶段 | 启动 Manager 线程（最多等 30 s 就绪） |
| `request_gpu(request_id=None, payload_ref=None, *, kind=None)` | `ON_DEMAND` 业务 Worker | 发布 GPU 请求，返回请求对象；不代表已获得 GPU |
| `wait_for_grant(request, timeout=None)` | `ON_DEMAND` 业务 Worker | 阻塞到该请求获批；返回前 `on_acquire` 已执行完毕 |
| `release(transition="", payload_ref=None)` | 当前 GPU 持有者 | 执行 `on_release`，选择下一个目标并发布 Grant；`payload_ref` 随 Grant 传递 |
| `occupy(request, transition="")` | 可选 | `wait_for_grant()` 和 `release()` 的上下文管理器简写 |
| `owns_gpu` | 任意 | 当前是否持卡 |
| `stop()` | Service 退出阶段 | 停止 Manager 线程 |

对 `SpmdService` 子类（GPU 引擎角色）而言，上述接线已在 `SpmdService._run_runtime` 里完成：回调直接绑定到 `engine.on_colocate_acquire / on_colocate_release`，Service 只需实现 `build_engine()` 与 `build_worker()`。

### Service 适配代码：显存释放与恢复回调

两个回调是 Service 唯一必须实现的显存边界：

| 回调 | 触发时机 | 必须完成的工作 |
| --- | --- | --- |
| `on_release(target)` | 当前 Service 持有 GPU，准备发布 Grant | 停止接收新任务，在安全边界停止计算，释放全部 GPU 显存；完成后 Manager 才发布 `source -> target` |
| `on_acquire(grant)` | 本 Service 收到指向自己的 Grant | 恢复显存和运行状态（可从 `grant.payload_ref` 取新权重）；回调返回后，等待该请求的 Worker 才会被唤醒 |

多卡引擎的回调要让**每个 rank** 同步改变驻留：`TitanEngine` 在 master 上把回调转成 SPMD 命令（`colocate_acquire` / `colocate_release`）广播给全体 rank。

```python
# TitanEngine（meshy/engine/titan.py）
def on_colocate_release(self, target: str) -> None:
    if self.world_size > 1 and self.is_master:
        self.submit_command("colocate_release")   # 全 rank offload_to_cpu
    else:
        self.offload()

def on_colocate_acquire(self, grant: GpuGrant) -> None:
    if self.world_size > 1 and self.is_master:
        self.submit_command("colocate_acquire")   # 全 rank restore_to_gpu
    else:
        self.restore()
```

Service 退出时调用 `self.colocation.stop()`。

### 业务调用代码：`ON_DEMAND` Service

Training 这类按需使用 GPU 的 Service 使用 `ON_DEMAND`（`meshy/worker/titan.py:TitanWorker.process_tq_batch`）：

```python
request = self.colocation.request_gpu(
    request_id=f"{self.engine.name}:window:{self.engine.step_index}",
)
self.colocation.wait_for_grant(request)
result = self.engine.step(samples, sync=True)
self.colocation.release(
    transition="train-step-complete",
    payload_ref=result.weights_path,      # 接手的 SGLang 从这里加载新权重
)
```

| 调用 | 含义 |
| --- | --- |
| `request_gpu()` | 向账本写入 `GpuRequest`。只表示“我需要 GPU”，不会立即获得 GPU。重试同一个逻辑任务时应复用同一个 `request_id`。 |
| `wait_for_grant(request)` | 阻塞直到该请求收到 `GpuGrant`，并在返回前执行本 Service 的 `on_acquire`。 |
| `release(transition=..., payload_ref=...)` | 执行 `on_release`，按排序键选择下一个请求并发布 Grant。 |

`request_id` 必须对应一个逻辑任务，而不是每次重试都生成新的随机 id。`transition` 只是调试和追踪信息；`payload_ref` 会被接手方消费。

`occupy()` 可以把等待和释放写成上下文管理器：

```python
request = self.colocation.request_gpu(request_id="scorer:step:42")
with self.colocation.occupy(request, transition="score-complete"):
    self.engine.score(samples)
```

### Service 适配代码：`FALLBACK` Service 的回调

Inference 这类默认工作角色使用 `FALLBACK`，不需要在业务代码里调用 `request_gpu()`；manager 会自动为它维持常驻请求。它只需提供两个回调（`meshy/engine/sglang.py:SGLangEngine`）：

```python
def on_colocate_release(self, target: str) -> None:
    self.release_for_colocate()      # pause(abort) → wait_until_idle → release(kv_cache, weights)

def on_colocate_acquire(self, grant: GpuGrant) -> None:
    weights = grant.payload_ref or (self.model_path if grant.sequence == 0 else None)
    if weights is None:
        raise RuntimeError("SGLang colocate acquire requires the latest checkpoint path in grant.payload_ref")
    self.restore_for_colocate(weights)   # resume(weights) → load → resume(kv) → continue
```

推理副本是多进程 TP 时，只有 ServiceGroup leader 持有 manager，它用**全组端点**构造的 `SGLangEngine` 代理整个组的 release / restore。

被 `pause_generation(abort)` 打断的 `/generate` 请求会返回部分输出，`SGLangEngine.generate` 自动带前缀续传，因此 rollout 侧不需要感知让出。

### 框架行为：不使用 Colocation 时的兼容模式

环外 Service 会获得 `NoopColocationManager`：`owns_gpu` 恒为 `True`，`wait_for_grant()` 立即返回，`release()` 不执行调度动作。因此同一套业务代码也可以运行在 disaggregate 拓扑下。

---

## 故障排除

### `colocation command 'release' timed out`

Manager 线程已经退出。`_apply_grant` 的协议校验失败（`sequence` 不连续、`source` 与上一任 `target` 不符、genesis 非法）会直接终止线程，调用方等到 30 秒超时。请检查该 Service 的日志中是否有更早的异常堆栈（`Colocation manager <service_id> failed`）。

线程退出时会唤醒所有等待中的 `wait_for_grant()`，它们抛出 `RuntimeError("colocation manager failed")` 而不是永久阻塞；`wait_for_grant()` 本身没有超时。

### Trainer 报 OOM，且 Inference 显存未释放

确认让出方的 `on_release` 回调执行完毕后才发布 Grant。若日志显示 Grant 的 `sequence` 已经推进但 Inference 侧没有对应的 `release_memory_occupation` 记录，说明回调抛出了异常。

### 令牌停在某个 ON_DEMAND Service 上不动

检查环中是否存在 `FALLBACK` 成员。全 `ON_DEMAND` 的环在所有请求耗尽后会停止流转。

### SGLang 恢复后报 `SGLang colocate acquire requires the latest checkpoint path in grant.payload_ref`

训练侧 `release` 时没有带 `payload_ref`。`TitanWorker` 在共卡模式下要求 `StepResult.weights_path` 非空；若你写了自定义 Worker，`release` 必须传 ckpt 路径。

---

## 设计说明

### 为什么用 TQ 账本而不是 HTTP 仲裁

Colocate 的本质是**物理互斥**。把状态放在 TQ 上的一张账本里，所有 manager 都能还原出相同的状态，grant `sequence` 严格连续使每次转移都可去重、可重放、可审计；新增角色只需要在 `ring` 中增加一项，不需要谁去调用谁的端点。

### 为什么是可变行而不是 append-only 事件流

一个请求从 `open` 到 `granted` 到 `closed` 是同一行的状态变化，原地更新让账本大小 = 请求数而不是事件数，也让“某请求是否已授权”成为一次行读取而非事件回放。代价是扫描必须 `force_fetch`，见「下一步」。

---

## 下一步

- **账本回收**——`close_request` 只把行标为 `closed`，`purge_request`（`clear_samples`）尚未被 manager 调用，账本随 step 数增长。TQ 的 `pre_alloc_sample_num` 下限为 1024，每个训练 step 至少产生 1 个 request 行。
- **多副本 ON_DEMAND 校验**——`is_colocation_leader` 按副本序号设置，非 leader 副本会获得 `NoopColocationManager`，因此不会申请令牌而直接训练。这对 Inference 是正确的（leader 通过全组端点代理），对多副本 Training 则不然。
- **Rollout 作为环成员**——`RolloutWorker` 已实现 `FALLBACK` 类请求，但 `RolloutServiceConfig.uses_gpu=False` 使它无法通过环校验，现有 recipe 里它始终拿到 Noop manager。

相关代码：

- `meshy/service/colocation.py` —— `ColocationRing` / `ColocationManager` 状态机、`issue_genesis`
- `meshy/transferqueue/colocation.py` —— `TQRequestLedgerTransport` 账本传输
- `meshy/service/topology.py` —— `_attach_colocations` 的拓扑校验
- `meshy/engine/titan.py` / `meshy/engine/sglang.py` —— 两种引擎的回调实现
- `meshy/worker/titan.py` —— `ON_DEMAND` 业务调用
- `tests/test_meshy_colocation.py`、`tests/test_weight_sync.py` —— 相关测试
