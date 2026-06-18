---
title: '集合通信入门'
date: 2026-06-18 00:00:00 +0800
permalink: /posts/collective-communication-primer/
categories: [Machine Learning]
tags: [distributed-training, collective-communication, PyTorch, NCCL, HCCL]
---

本文面向刚接触多卡训练和分布式通信的读者，用具体例子解释 collective communication。
这里的内容刻意保持 backend-agnostic：先建立 rank、process group、communicator、
stream 和常见 collective 的心智模型，再用 NCCL 和 HCCL 展示它们在真实后端中的
形状。

这不是对任何特定项目或框架 API 稳定性的承诺。不同后端在 C API、datatype enum、
stream handle、communicator 创建流程、错误处理和非均匀 shape 支持上都会有差异。
阅读本文时，请把代码块当作简化示例，把具体约束以对应后端文档为准。

## 一屏概要

- 集合通信是多个训练或推理进程把张量作为一次协同操作进行交换的机制。
- PyTorch 把每个参与进程称为一个 rank。一组 rank 构成一个 process group。
- backend 是把 PyTorch 集合通信操作转换为设备通信的具体实现，例如 CUDA 上常见的
  `nccl`，Ascend 环境中的 `hccl`。
- Store 或 hostfile 这类 control plane 组件用于在设备通信开始前交换少量 metadata。
- communicator 是 data plane 对象，负责连接一组 rank 并提交真实的 tensor payload
  通信。
- 大多数通信 API 都是异步的：host 函数返回通常只表示通信已经被提交到设备 stream，
  不一定表示 payload 已经完成传输。
- 理解 communication pattern 时，把“谁和谁通信”与“tensor payload 发生什么变化”
  分开会更清楚。
- 分布式训练还需要启动、rendezvous、rank 分配、超时处理、stream/event 同步、内存
  生命周期保护和调试工具。

最有用的首次阅读路径是：

```text
launcher starts N processes
  -> rendezvous or hostfile assigns ranks
  -> init_process_group creates a process group backend
  -> backend creates or reuses communicators
  -> collective call submits communication work on device streams
  -> work handle, stream sync, or framework logic tracks completion
```

## Training Workflow Context

如果先看一个分布式训练 step 的形状，集合通信会更容易定位。

```text
initialize model and optimizer state
  -> optionally broadcast initial parameters
  -> forward pass
  -> loss computation
  -> backward pass produces gradients
  -> communicate gradients, activations, or tensor shards
  -> optimizer updates parameters
  -> repeat
```

训练循环也可以理解为一系列通信压力点：

| Stage | What happens locally | Communication relevance |
| ---- | ------------ | ------------ |
| Initialization | 设置参数、buffer 和优化器状态。 | 副本式训练可能从一个 root rank 广播初始状态。 |
| Forward pass | 每个 rank 为本地输入计算激活。 | 模型并行布局可能通信激活或 shard。 |
| Loss computation | 将 prediction 与 target 比较。 | 通常是本地操作，除非 logits 或 metrics 被切分。 |
| Backward pass | 根据保存的 activation 产生 gradient。 | DDP、FSDP、TP 和 PP 会加入不同的 communication point。 |
| Optimizer step | 根据 gradient 更新 parameter。 | 各副本必须在 gradient 或 parameter shard 上达成一致。 |
| Iteration | 下一批数据重复同样的 dependency pattern。 | 各 rank 的 collective order 必须保持一致。 |

通信方式取决于并行策略：

- 数据并行通常在每个 rank 上复制模型，把输入 batch 切分到各 rank，然后在每个副本
  执行同样的优化器更新前用 `all_reduce` 聚合梯度。
- 张量并行或模型并行把一个逻辑模型操作切分到多个 rank 上，因此 `all_gather`、
  `reduce_scatter` 和 `all_to_all` 等 collective 可能出现在前向或反向传播内部。
- 流水线并行把层切分为多个 stage。它通常需要 point-to-point `send` / `recv`，
  在相邻 stage 之间传递激活和梯度。
- Expert 或 MoE 并行会把 token 路由到不同 rank。它通常使用 `all_to_all` 或
  非均匀变体来重新排列可变大小的 token 分组。

因此，集合通信位于模型代码之下、设备互联之上：模型代码描述数学依赖，通信 backend
执行所需的数据移动。

### PyTorch DistributedDataParallel Background

PyTorch 里常见的多卡训练方式是 `DistributedDataParallel`，简称 DDP。DDP 的基本
背景是：每张卡通常对应一个独立 Python process，每个 process 有自己的 rank，并在
自己的 device 上持有一份完整 model replica。输入 dataset 会通过
`DistributedSampler` 或等价逻辑切成 per-rank mini-batch。

这和单进程多卡的 `DataParallel` 不同。`DataParallel` 在一个 Python process 里
scatter input、replicate module、gather output，容易受 Python 调度和单进程瓶颈
影响。DDP 把每张卡放到独立 process，forward/backward 主要在本地 device 上执行，
跨卡只同步必要的 gradient，因此是 PyTorch 推荐的主流 data-parallel training
方式。

最小化的 DDP mental model 是：

```text
torchrun starts N processes
  -> each process gets one rank and one local device
  -> init_process_group creates a distributed process group
  -> each rank builds the same model replica
  -> DistributedDataParallel wraps the model and installs autograd hooks
  -> each rank runs forward/loss/backward on its local mini-batch
  -> DDP all-reduces gradients so every replica sees the same averaged gradients
  -> each rank runs optimizer.step() locally with identical gradients
```

一个 CUDA/NCCL 环境中常见的 PyTorch 训练 skeleton 如下：

```python
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)

model = MyModel().to(device)
model = DDP(model, device_ids=[local_rank])

for input, target in loader:
    input = input.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

Ascend/HCCL 环境在 PyTorch 层面的形状类似，只是 device module 和 backend 名称会
随框架适配层变化。常见模式是使用 `backend="hccl"`，每个 process 绑定一个本地
NPU：

```python
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch_npu

dist.init_process_group(backend="hccl")
local_rank = int(os.environ["LOCAL_RANK"])
device = f"npu:{local_rank}"
torch.npu.set_device(device)

model = MyModel().to(device)
model = DDP(model, device_ids=[local_rank])
```

上面代码里，`loss.backward()` 是容易误解的地方。DDP 并不是把一个 rank 上的 loss
“传回”所有卡。每个 rank 都有自己的 local input、local target、local output 和
local loss。`backward()` 从本地 loss 出发，先在本地 model replica 上计算 local
gradients；当某些 parameter gradients 准备好后，DDP 的 autograd hook 会触发
gradient communication。

### Two-Card DDP Loss And Backward Example

假设有 2 张卡，`world_size = 2`：

| Rank | Device | Local mini-batch | Local loss | Local gradients before DDP sync |
| ---- | ------ | ---------------- | ---------- | ------------------------------- |
| rank 0 | device 0 | samples `[0..31]` | `loss_0` | `grad_0` |
| rank 1 | device 1 | samples `[32..63]` | `loss_1` | `grad_1` |

每个 rank 的 forward 和 loss computation 都是本地发生的：

```text
rank 0: output_0 = model_0(input_0), loss_0 = criterion(output_0, target_0)
rank 1: output_1 = model_1(input_1), loss_1 = criterion(output_1, target_1)
```

随后两个 rank 都调用 `loss.backward()`。本地 autograd 先计算各自的 gradient：

```text
rank 0: grad_0 = d(loss_0) / d(parameter_0)
rank 1: grad_1 = d(loss_1) / d(parameter_1)
```

DDP 接下来对同一个 parameter 的 gradient 执行 `all_reduce`：

```text
all_reduce(sum): grad_sum = grad_0 + grad_1
average:         grad_avg = grad_sum / world_size

rank 0 parameter.grad = grad_avg
rank 1 parameter.grad = grad_avg
```

这样 `optimizer.step()` 虽然在每个 rank 本地执行，但每个 model replica 看到的是
同一份 averaged gradient，因此参数更新保持一致。严格说，跨卡同步的对象是
gradient，不是 loss。loss 可以单独用于 logging；如果希望打印全局平均 loss，可以
额外对 loss value 做一次 `all_reduce`，但这通常不参与 parameter update。

这个流程也解释了 batch size 的语义。如果每张卡的 local batch size 是 32，
`world_size = 2`，那么一次同步训练 step 的 effective global batch size 是 64。
多数 loss function 默认先在每个 rank 的 local mini-batch 内做 mean reduction，
然后 DDP 再平均 gradient；这通常等价于对 global batch 求平均梯度。若使用 sum
reduction、自定义 loss scaling、gradient accumulation 或不均匀 batch，需要额外
确认 scaling 是否符合预期。

## Examples First

下面的 examples 先不关心具体 backend，只看每个 rank 上的 tensor 如何变化。读完
这些例子后，再看后面的 term 和 backend lifecycle 会更容易。

### Example 1: DDP all_reduce With One Scalar Parameter

先看一个极小的 DDP 例子。假设模型只有一个 scalar parameter `w`，模型是
`prediction = w * x`。两个 rank 的初始参数都是 `w = 1.0`，learning rate 是 `0.1`。

| Rank | Local sample | Target | Local prediction | Local loss |
| ---- | ------------ | ------ | ---------------- | ---------- |
| rank 0 | `x = 1` | `target = 2` | `1 * 1 = 1` | `(1 - 2)^2 = 1` |
| rank 1 | `x = 3` | `target = 6` | `1 * 3 = 3` | `(3 - 6)^2 = 9` |

每个 rank 的 loss 是本地算出来的。DDP 不会把 `loss_0` 或 `loss_1` 发送给其他
rank 来做 backward。每个 rank 先根据自己的 local loss 算 local gradient：

```text
loss = (w * x - target)^2
d(loss) / d(w) = 2 * (w * x - target) * x

rank 0: grad_0 = 2 * (1 * 1 - 2) * 1 = -2
rank 1: grad_1 = 2 * (1 * 3 - 6) * 3 = -18
```

然后 DDP 对 `w.grad` 执行 `all_reduce(sum)`，再除以 `world_size`：

```text
all_reduce(sum): grad_sum = -2 + -18 = -20
average:         grad_avg = -20 / 2 = -10

rank 0 sees w.grad = -10
rank 1 sees w.grad = -10
```

最后每个 rank 本地执行同样的 `optimizer.step()`：

```text
w_new = w_old - learning_rate * grad_avg
      = 1.0 - 0.1 * (-10)
      = 2.0
```

两个 rank 没有互相发送 loss，但它们通过 gradient `all_reduce` 得到了相同的
`w.grad`，所以本地 optimizer update 后的 `w` 仍然一致。

### Example 2: broadcast Initial Parameters

`broadcast` 常用于初始化或加载 checkpoint 后同步模型参数。假设 rank 0 从
checkpoint 加载了参数，其他 rank 的参数还没同步：

| Rank | Before `broadcast` |
| ---- | ------------------ |
| rank 0 | `weight = [0.2, -1.0, 3.0]` |
| rank 1 | `weight = [?, ?, ?]` |
| rank 2 | `weight = [?, ?, ?]` |
| rank 3 | `weight = [?, ?, ?]` |

执行：

```text
broadcast(tensor=weight, root=0)
```

结果是：

| Rank | After `broadcast(root=0)` |
| ---- | ------------------------- |
| rank 0 | `weight = [0.2, -1.0, 3.0]` |
| rank 1 | `weight = [0.2, -1.0, 3.0]` |
| rank 2 | `weight = [0.2, -1.0, 3.0]` |
| rank 3 | `weight = [0.2, -1.0, 3.0]` |

这个 operation 的重点是“root rank 的 tensor 被复制给所有 rank”。它不做 sum、mean
或其他 reduction。

### Example 3: all_gather Rebuilds Sharded Tensor

`all_gather` 常用于把每个 rank 上的一段 shard 收集起来，让每个 rank 都看到完整
tensor。假设 2 个 rank 各自持有一半 hidden state：

| Rank | Before `all_gather` |
| ---- | ------------------- |
| rank 0 | `shard = [10, 11]` |
| rank 1 | `shard = [20, 21]` |

执行：

```text
all_gather([rank0_shard, rank1_shard])
```

结果是：

| Rank | After `all_gather` |
| ---- | ------------------ |
| rank 0 | `full = [10, 11, 20, 21]` |
| rank 1 | `full = [10, 11, 20, 21]` |

这个 operation 不合并数值，只是把各 rank 的 shard concatenate/list 到每个 rank。
Tensor parallelism 里经常需要这样的重组步骤。

同样的 pattern 放到 4 个 rank 上也一样。初始状态是每个 rank 只有自己的 chunk：

| Rank | Local data before `all_gather` |
| ---- | ------------------------------ |
| rank 0 | `[D0]` |
| rank 1 | `[D1]` |
| rank 2 | `[D2]` |
| rank 3 | `[D3]` |

通信完成后，每个 rank 都得到相同顺序的完整结果：

| Rank | Data after `all_gather` |
| ---- | ----------------------- |
| rank 0 | `[D0, D1, D2, D3]` |
| rank 1 | `[D0, D1, D2, D3]` |
| rank 2 | `[D0, D1, D2, D3]` |
| rank 3 | `[D0, D1, D2, D3]` |

所以 `all_gather` 的关键词是 shard reconstruction：收集、拼接、复制到所有 rank。
它不做 sum/mean，因此不要把它和 DDP gradient averaging 混在一起。DDP 同步
gradient 的核心 collective 通常是 `all_reduce`。

### Example 4: reduce_scatter Reduces Then Keeps One Shard

`reduce_scatter` 可以理解成先做 `all_reduce`，再把结果切开，每个 rank 只保留自己
的 shard。假设 2 个 rank 都有一个 length-4 tensor：

| Rank | Local tensor before `reduce_scatter(sum)` |
| ---- | ----------------------------------------- |
| rank 0 | `[1, 2, 3, 4]` |
| rank 1 | `[10, 20, 30, 40]` |

先按元素 sum：

```text
reduced = [1 + 10, 2 + 20, 3 + 30, 4 + 40]
        = [11, 22, 33, 44]
```

然后 scatter 成 2 份：

| Rank | After `reduce_scatter(sum)` |
| ---- | --------------------------- |
| rank 0 | `[11, 22]` |
| rank 1 | `[33, 44]` |

这个 pattern 在 tensor parallelism、optimizer state sharding 和部分大模型训练流程
里很常见，因为它既完成 reduction，又避免每个 rank 都保留完整结果。

### Example 5: all_to_all Routes Token Shards

`all_to_all` 常用于 MoE 或 layout transpose：每个 rank 都有一些 shard 要发给不同
destination rank。假设 2 个 rank 上有 6 个 token，路由结果如下：

| Source rank | Tokens kept for rank 0 | Tokens sent to rank 1 |
| ----------- | ---------------------- | --------------------- |
| rank 0 | `[t0, t2]` | `[t1]` |
| rank 1 | `[t4]` | `[t3, t5]` |

执行 `all_to_all` 后，每个 destination rank 收到属于自己的 token：

| Destination rank | After `all_to_all` |
| ---------------- | ------------------ |
| rank 0 | `[t0, t2, t4]` |
| rank 1 | `[t1, t3, t5]` |

这个 operation 的重点不是 sum，也不是 broadcast，而是 reshuffle。每个 rank 同时发送
和接收不同切片，因此 tensor split size、destination order 和 shape metadata 必须
在所有 rank 上一致。

## Communication Operation Diagrams

下面的图例使用 rank、tensor block 和 arrow 展示常见 collective operation。这些图
是为了帮助理解 communication pattern，不代表 HCCL 或 NCCL 的真实传输算法。

阅读这些图时可以分成两层：

- State view：operation 前后，每个 rank 持有什么 tensor 或 shard。
- Flow view：通信过程中，哪些 rank 需要把 payload 发给哪些 peer。

很多 collective 可以用更简单的 operation 组合来建立 mental model。例如
`all_gather` 可以理解成“每个 rank 都参与 gather，然后每个 rank 都拿到 gather 后的
完整结果”；也可以粗略想成 gather + broadcast-style replication。`all_reduce` 则
常被分解成 `reduce_scatter` + `all_gather` 来理解。真实 backend 可能使用 ring、
tree、分层算法或硬件专用路径，不一定真的先集中到某一个 root rank 再广播，也不一定
按文档图例中的阶段顺序提交底层传输。

### Point-to-Point: send / recv

`send` / `recv` 是 one-to-one communication：一个 source rank 把 tensor 发给一个
destination rank。它不是“所有 rank 一起参与”的 collective，但在 pipeline
parallelism 和自定义 distributed protocol 里很常见。

```text
before:
rank 0 source:      A
rank 1 destination: empty recv buffer

operation:
rank 0 calls send(A, dst=1)
rank 1 calls recv(buffer, src=0)

after:
rank 0: A
rank 1: A
```

这里的重点是 match：source、destination、tensor shape、dtype 和调用顺序必须对得
上。否则一个 rank 可能已经 send，而另一个 rank 没有进入对应的 recv。

![send recv point-to-point communication](/assets/collective-communication/send-recv.svg)

常见场景：pipeline parallelism 中相邻 stage 之间传递 activation 或 gradient。

### One-to-Many: broadcast

`broadcast` 是 one-to-many communication：root rank 的完整 tensor 被复制到所有
rank。只有 root rank 提供有效输入，其他 rank 通常提供同 shape 的接收 buffer。

```text
before:
rank 0 root: W
rank 1:      empty
rank 2:      empty
rank 3:      empty

after broadcast(root=0):
rank 0: W
rank 1: W
rank 2: W
rank 3: W
```

这个 operation 不切片，也不做 sum；它只是把 root payload 复制到 group 内所有 rank。
真实 backend 可能使用 tree-style forwarding，不一定由 root 直接发给每个 peer。

![broadcast one-to-many communication](/assets/collective-communication/broadcast.svg)

常见场景：rank 0 初始化或加载 checkpoint 后，把 model parameters 同步到所有
model replica。

### One-to-Many: scatter

`scatter` 也是 one-to-many communication，但 root rank 发送的是不同 shard。可以把
它理解成 root 先把一个大 buffer 按 rank order 切开，再把第 `Y` 段发给 rank `Y`。

```text
before:
rank 0 root: [A0, A1, A2, A3]

after scatter(root=0):
rank 0: A0
rank 1: A1
rank 2: A2
rank 3: A3
```

如果每个输出 shard 有 `count` 个元素，rank `Y` 的本地输出可以理解为来自 root input
中的 `[Y * count + i]` 位置。这里的重点是 split 和 destination order。

![scatter one-to-many communication](/assets/collective-communication/scatter.svg)

常见场景：把一个 tensor、batch 或 work list 切成 per-rank shard。

### Many-to-One: gather

`gather` 是 many-to-one communication：每个 rank 发送一个 shard，root rank 收集
所有 shard。它是 `scatter` 的反向操作，但只有 root rank 得到完整结果。

```text
before:
rank 0 root: D0
rank 1:      D1
rank 2:      D2
rank 3:      D3

after gather(root=0):
rank 0 root: [D0, D1, D2, D3]
rank 1:      no gathered output
rank 2:      no gathered output
rank 3:      no gathered output
```

root 的 output buffer 通常按 rank order 组装：rank `Y` 的输入放到
`out[Y * count + i]`。这也是 `gather` 和 `all_gather` 的核心区别：前者只有 root
收到完整 buffer，后者每个 rank 都收到完整 buffer。

![gather many-to-one communication](/assets/collective-communication/gather.svg)

常见场景：集中收集 validation result、debug tensor 或较小的 metadata tensor。

### Many-to-One: reduce

`reduce` 也是 many-to-one communication，但 root rank 收到的是 reduction 后的
结果，而不是简单 concatenate。所有 rank 输入同 shape buffer，backend 按元素应用
reduction operation，例如 `sum`、`max` 或 `min`。

```text
rank 0 root input: [A0, A1]
rank 1 input:      [B0, B1]
rank 2 input:      [C0, C1]

after reduce(sum, root=0):
rank 0 root output: [A0+B0+C0, A1+B1+C1]
rank 1:             no reduced output
rank 2:             no reduced output
```

这个 operation 和 `gather` 的通信方向类似，都是 many-to-one；区别是 `gather`
保留每个 shard 的原值和顺序，而 `reduce` 把同位置元素合并成一个 reduced result。

![reduce many-to-one communication](/assets/collective-communication/reduce.svg)

常见场景：只需要在一个 rank 上得到全局 metric 或统计值。

### Many-to-Many: all_gather

`all_gather` 是 gather 的 all-rank 版本：每个 rank 从“只持有自己的 shard”变成
“持有所有 rank 的 shard”。如果有 4 个 rank，初始状态是：

```text
rank 0: [D0]
rank 1: [D1]
rank 2: [D2]
rank 3: [D3]
```

通信阶段里，每个 rank 会把自己的 local chunk 提供给其他 rank，并接收其他 rank
的 chunk。完成后每个 rank 得到相同顺序的 combined tensor：

```text
rank 0: [D0, D1, D2, D3]
rank 1: [D0, D1, D2, D3]
rank 2: [D0, D1, D2, D3]
rank 3: [D0, D1, D2, D3]
```

![all gather many-to-many communication](/assets/collective-communication/all-gather.svg)

常见场景：tensor parallelism 中把 sharded hidden state 或 partial result 重组成
完整 tensor；FSDP 或 ZeRO 风格的参数分片也可能在需要完整参数时使用 `all_gather`。
注意 DDP gradient synchronization 通常是 `all_reduce`，不是 `all_gather`。

### Many-to-Many: all_reduce

`all_reduce` 是 reduce 的 all-rank 版本：每个 rank 输入一个同 shape 的 buffer，
所有 buffer 按元素做 reduction，然后每个 rank 都得到完全相同的 reduced result。
以 3 个 rank 的 `sum` 为例：

```text
rank 0 input: [A0, A1, A2]
rank 1 input: [B0, B1, B2]
rank 2 input: [C0, C1, C2]

global sum:  [A0+B0+C0, A1+B1+C1, A2+B2+C2]

rank 0 output: [A0+B0+C0, A1+B1+C1, A2+B2+C2]
rank 1 output: [A0+B0+C0, A1+B1+C1, A2+B2+C2]
rank 2 output: [A0+B0+C0, A1+B1+C1, A2+B2+C2]
```

算法上，`all_reduce` 经常可以理解为 `reduce_scatter` + `all_gather`：第一阶段
每个 rank 得到一段已经 reduce 完成的 shard，第二阶段再把这些 reduced shard
all-gather 到所有 rank。NCCL 等 backend 常见的 ring all-reduce 就利用了这种分解
来提高 bandwidth utilization；实际 backend 也可能根据 topology、tensor size 或
硬件能力选择 tree、hierarchical 或专用算法。

![all reduce many-to-many communication](/assets/collective-communication/all-reduce.svg)

常见场景：DDP gradient synchronization。实际训练里通常还会除以 `world_size` 得到
averaged gradient。也就是说，底层 collective 可能先产生 summed gradient，framework
或 communication hook 再做 scaling。

### Many-to-Many: reduce_scatter

`reduce_scatter` 可以理解为：先 reduce，再 scatter reduced tensor。每个 rank 只
保留最终结果的一段 shard。以 3 个 rank 的 `sum` 为例：

```text
rank 0 input: [A0, A1, A2]
rank 1 input: [B0, B1, B2]
rank 2 input: [C0, C1, C2]

reduced full result:
[S0, S1, S2]
where S0=A0+B0+C0, S1=A1+B1+C1, S2=A2+B2+C2

after reduce_scatter(sum):
rank 0: S0
rank 1: S1
rank 2: S2
```

它和 `all_reduce` 的前半段 mental model 一致：先得到 reduced shards；区别是
`reduce_scatter` 不会再 all-gather 回完整结果，所以每个 rank 只保留自己的 reduced
shard。

![reduce scatter many-to-many communication](/assets/collective-communication/reduce-scatter.svg)

常见场景：tensor parallelism 和 sharded optimizer 中减少每个 rank 保留的结果大小。

### Many-to-Many: all_to_all

`all_to_all` 是 reshuffle：每个 rank 把不同 shard 发给不同 destination rank，同时
也从所有 source rank 接收属于自己的 shard。

```text
before: each row is one source rank, suffix is destination rank
rank 0: [A0, A1, A2]
rank 1: [B0, B1, B2]
rank 2: [C0, C1, C2]

after all_to_all:
rank 0: [A0, B0, C0]
rank 1: [A1, B1, C1]
rank 2: [A2, B2, C2]
```

它不做 reduction，也不是 broadcast。可以把它看作 distributed layout transpose：
输入按 source rank 分组，输出按 destination rank 分组。fixed-size `all_to_all`
要求每个 rank 发给各 destination 的 shard size 一致；非均匀大小需要 `all_to_allv`
或 split-size 变体。

![all to all many-to-many communication](/assets/collective-communication/all-to-all.svg)

常见场景：MoE token routing、layout transpose、sequence/tensor parallel 中的维度
重排。

### Many-to-Many: all_to_allv

`all_to_allv` 是 variable-size all-to-all。每个 source rank 发给不同 destination
rank 的 shard 大小可以不同。

```text
rank 0 sends: dst0 -> [A0, A1], dst1 -> [A2],     dst2 -> [A3]
rank 1 sends: dst0 -> [B0],     dst1 -> [],       dst2 -> [B1, B2]
rank 2 sends: dst0 -> [C0, C1], dst1 -> [C2],     dst2 -> [C3]

after all_to_allv:
rank 0: [A0, A1, B0, C0, C1]
rank 1: [A2, C2]
rank 2: [A3, B1, B2, C3]
```

和 fixed-size `all_to_all` 相比，`all_to_allv` 的核心额外约束是 split-size metadata：
每个 rank 必须就发送和接收长度达成一致，否则 destination 端无法正确切分 buffer。

![all to all v variable size communication](/assets/collective-communication/all-to-allv.svg)

常见场景：MoE 中不同 expert 收到的 token 数量不均匀。它比 fixed-size `all_to_all`
更灵活，但也更依赖准确的 split size metadata。

## Core Terms

| Term | Distributed meaning | Where it appears |
| ---- | ------------------ | ---------------- |
| Rank | 一个参与进程。rank id 通常是 `0..world_size-1`。 | PyTorch process group、NCCL communicator、HCCL/MPI launch。 |
| Local rank | 单台 host 或 node 内部的 rank 索引。 | launcher 提供的 `LOCAL_RANK`，通常用于选择本地 device。 |
| World size | 参与 rank 的数量。 | `WORLD_SIZE`、`nranks`、MPI `-n`。 |
| Node | 一台 host，可以运行一个或多个 local rank。 | launcher、hostfile、rendezvous 和网络拓扑。 |
| Process group | 一组一起执行 collective 的 rank 子集。 | PyTorch `dist.init_process_group` 和 subgroup。 |
| Backend | PyTorch c10d 操作背后的具体实现。 | `nccl`、`hccl`、`gloo` 等。 |
| Store | 用于 rank 元数据的小型分布式键值服务。 | PyTorch c10d rendezvous、TCPStore、FileStore。 |
| Hostfile | MPI 风格的节点与 slot 描述。 | HCCL 性能测试中常见的 `mpirun -f hostfile`。 |
| Rendezvous | 启动协议，为每个进程提供 Store、rank 和 world size。 | `torchrun`、elastic launcher、集群调度系统。 |
| Communicator | communication library 用来连接 rank 的 runtime object。 | NCCL `ncclComm_t`、HCCL communicator。 |
| Root rank | 在某些操作中提供或接收数据的特殊 rank。 | `broadcast`、`scatter`、`gather` 和 `reduce` 选项。 |
| Reduction operation | 用来合并值的 function，例如 sum、max、min 或 product。 | `ReduceOp`、`ncclRedOp_t`、HCCL test 的 `-o` 参数。 |
| Shard | 一个更大逻辑张量的切片。 | `all_gather`、`reduce_scatter`、张量并行和优化器状态切分。 |
| Collective | group 中所有 rank 按相同顺序调用的一次操作。 | `all_reduce`、`broadcast`、`all_gather`、`reduce_scatter` 等。 |
| Point-to-point | 一个 rank 直接向另一个 rank 发送数据。 | `send`、`recv`、NCCL `ncclSend` / `ncclRecv`。 |
| Work | 通信调用返回的 asynchronous handle。 | PyTorch distributed API 中的 async collective。 |
| Stream | 设备侧执行队列。 | CUDA stream、NPU stream 等。 |
| Event | 用于排序 stream 和查询完成状态的设备侧完成标记。 | CUDA event、NPU event 等。 |

## Communication Pattern Taxonomy

Ascend HCCL 资料和 NCCL 文档都会把 communication pattern 拆成 one-to-one、
one-to-many、many-to-one、many-to-many 等类别。即使具体 API name 不同，这套
taxonomy 也适合阅读大多数 collective backend。

| Pattern | Basic idea | Typical operations | Payload behavior |
| ---- | -------- | -------- | -------- |
| One-to-one | 一个 rank 向一个 peer 发送数据。 | `send`、`recv` | 把 tensor 从 sender 复制到 receiver。 |
| One-to-many | 一个 root rank 分发数据。 | `broadcast`、`scatter` | 复制完整 tensor，或切成 shard。 |
| Many-to-one | 多个 rank 向一个 root rank 发送数据。 | `gather`、`reduce` | concatenate shard，或在 root 上做 reduce。 |
| Many-to-many | 每个 rank 都和其他 rank 交换数据。 | `all_reduce`、`all_gather`、`reduce_scatter`、`all_to_all` | reduce、concatenate、split 或 reshuffle shard。 |

另一个同样重要的分类方式是看张量内容发生了什么变化。

- Copy operation 保留原值，例如 `send`、`recv` 和 `broadcast`。
- Split operation 把一个 logical tensor 拆成 per-rank shard，例如 `scatter`。
- Gather operation 拼接或列出各 rank 的 shard，例如 `gather` 和 `all_gather`。
- Reduce operation 把多个 rank 的值合并，例如 `reduce`、`all_reduce` 和
  `reduce_scatter`。
- Shuffle operation 把属于特定 rank 的 slice 移动到 destination rank，例如
  `all_to_all`。

## API Vocabulary Cross-Reference

在 naming 层面，NCCL、Ascend HCCL 和 PyTorch c10d 大体共享同一套 collective
vocabulary。具体 C function name、enum value、stream handle、communicator creation
path 和支持的 tensor layout 是 backend-specific 的，所以应把下表看作 concept
cross-reference，而不是 ABI contract。

| Concept | PyTorch/c10d vocabulary | NCCL-style API name | Ascend/HCCL-style API name |
| ---- | ----------------- | ------------------ | ------------------------- |
| Point-to-point send | `send` | `ncclSend` | `HcclSend` |
| Point-to-point recv | `recv` | `ncclRecv` | `HcclRecv` |
| Broadcast | `broadcast` | `ncclBroadcast` | `HcclBroadcast` |
| Scatter | `scatter` | `ncclScatter` | `HcclScatter` |
| Gather | `gather` | `ncclGather` | `HcclGather` |
| Reduce to root | `reduce` | `ncclReduce` | `HcclReduce` |
| Reduce then shard | `reduce_scatter` | `ncclReduceScatter` | `HcclReduceScatter` |
| Gather to all ranks | `all_gather` | `ncclAllGather` | `HcclAllGather` |
| Reduce to all ranks | `all_reduce` | `ncclAllReduce` | `HcclAllReduce` |
| Even all-to-all | `all_to_all` / `alltoall_base` | `ncclAlltoAll` | `HcclAlltoAll` |
| Uneven all-to-all | split-size all-to-all | send/recv group or backend-specific split/count support | `HcclAlltoAllV` |

并非每个 framework backend 都以同样名称暴露每个 operation，也不一定提供相同的 tensor
layout contract。例如 NCCL 文档中 `Scatter`、`Gather` 和 `AlltoAll` 是 collective
operation；在某些框架层，用户也可能用 point-to-point `send` / `recv` group 拼出等价
pattern。

## Common Collective Operations

具体 tensor layout 取决于模型，但 communication pattern 在不同 backend 之间是稳定
的。

| Operation | Input on each rank | Output on each rank | Common use |
| ---- | ------------------ | ------------------ | -------- |
| `send` / `recv` | 一个 rank 向一个 peer 发送张量。 | 接收方得到该张量。 | 流水线并行或自定义协议。 |
| `broadcast` | root rank 拥有数据，其他 rank 可能只有占位张量。 | 每个 rank 都得到 root 数据。 | 从 rank 0 发送模型权重或初始化状态。 |
| `scatter` | root rank 拥有列表，或拥有切成 shard 的张量。 | 每个 rank 接收一个 shard。 | 切分初始数据、参数或工作项。 |
| `gather` | 每个 rank 拥有一个 shard。 | root rank 接收所有 shard。 | 集中收集结果。 |
| `reduce` | 每个 rank 拥有一个张量。 | 一个 root rank 得到 reduce 后的结果。 | 集中收集 metric 或结果。 |
| `all_reduce` | 每个 rank 拥有一个张量。 | 每个 rank 都得到 reduce 后的张量，例如 sum。 | DDP 梯度平均。 |
| `all_gather` | 每个 rank 拥有一个 shard。 | 每个 rank 都得到拼接后或列表形式的全部 shard。 | 重建一个被切分的 activation、parameter 或 result。 |
| `reduce_scatter` | 每个 rank 拥有完整输入或分片输入用于 reduce。 | 每个 rank 得到一个 reduce 后的 shard。 | 张量并行和分片优化器流程。 |
| `all_to_all` | 每个 rank 拥有要发往其他 rank 的 shard。 | 每个 rank 接收发往自己的 shard。 | MoE token 路由或布局转置。 |
| `all_to_allv` | 每个 rank 拥有要发往其他 rank 的可变大小 shard。 | 每个 rank 接收可变大小的目标 shard。 | 非均匀 MoE token 路由或稀疏交换。 |
| `barrier` | 通常没有有意义的 tensor payload。 | 所有 rank 都到达后才继续执行。 | Synchronization point。 |

## NCCL Examples

NVIDIA NCCL 是许多 PyTorch 用户最熟悉的 GPU collective backend。NCCL repository
把它描述为用于 GPU 之间通信的优化 primitives，支持 `all-reduce`、`all-gather`、
`reduce`、`broadcast`、`reduce-scatter` 以及基于 send/receive 的通信 pattern。
NCCL 文档中的几个要点很适合建立 backend mental model：

- communicator 中的每个 rank 需要绑定到固定 CUDA device。
- 多进程场景通常先由一个进程调用 `ncclGetUniqueId()`，再通过 MPI、socket 或其他
  CPU 通信方式把 unique id 分发给所有参与者。
- `ncclCommInitRank()` 用 unique id、rank 数量和当前 rank 创建 `ncclComm_t`。
- collective 必须由每个 rank 用相同 count 和 datatype 参与，否则可能 hang、crash
  或产生数据损坏。
- NCCL collective 的最后一个参数通常是 CUDA stream；调用返回表示 operation 已经
  enqueue 到该 stream 或返回错误，真实通信随后在 device 上异步执行。
- `ncclGroupStart()` / `ncclGroupEnd()` 可用于一个线程管理多个 GPU、聚合多个通信
  operation，或把多组 `ncclSend` / `ncclRecv` 融合成复杂 pattern。

### NCCL Communicator Bootstrap

下面是多进程场景里的常见 communicator 创建形状。`broadcast_id_to_all_ranks` 可以是
MPI broadcast、socket、PyTorch Store 或任意 host-side rendezvous 机制。

```c
ncclUniqueId id;
ncclComm_t comm;

if (rank == 0) {
  ncclGetUniqueId(&id);
}

broadcast_id_to_all_ranks(&id, sizeof(id));

cudaSetDevice(local_device);
ncclCommInitRank(&comm, world_size, id, rank);
```

这个片段体现了 control plane 和 data plane 的分工：unique id 的分发是 control
plane，`ncclComm_t` 创建完成后的 tensor movement 是 data plane。

### NCCL all_reduce For A Gradient Bucket

DDP 的 gradient bucket 可以看成一段连续 buffer。假设每个 rank 都有 `bucket_in`，
希望得到所有 rank 的 sum：

```c
ncclAllReduce(
    bucket_in,
    bucket_out,
    bucket_count,
    ncclFloat32,
    ncclSum,
    comm,
    stream);
```

如果 framework 需要 averaged gradient，通常还要把 `bucket_out` 除以 `world_size`。
这个除法可能由框架通信 hook、optimizer 前的 tensor op，或 fused kernel 完成。

### NCCL all_gather For Tensor Parallel Shards

假设每个 rank 有 `shard_count` 个 FP16 元素，想按 rank order 重建完整 buffer：

```c
ncclAllGather(
    local_shard,
    full_buffer,
    shard_count,
    ncclFloat16,
    comm,
    stream);
```

完成后，`full_buffer` 的逻辑布局是：

```text
[rank0 shard][rank1 shard]...[rank(world_size-1) shard]
```

rank order 很重要。如果 rank 到 device 的映射不同，物理设备位置可能变化，但 buffer
里的逻辑顺序仍然按 rank id 理解。

### NCCL reduce_scatter As The First Half Of all_reduce

`reduce_scatter` 常见于 ZeRO/FSDP 风格的 shard 流程，也可以作为理解 all-reduce 的
第一阶段：

```c
ncclReduceScatter(
    full_grad_input,
    local_reduced_shard,
    shard_count,
    ncclFloat32,
    ncclSum,
    comm,
    stream);
```

如果 `world_size = 4`，每个 rank 最终只保留 `shard_count` 个 reduced 元素。相比
`all_reduce`，它减少了每个 rank 的结果驻留内存。

### NCCL Grouped P2P For all_to_all-style Exchange

NCCL 的 two-sided point-to-point API 要求一侧 `ncclSend` 与另一侧 `ncclRecv` 匹配。
多个 send/recv 可以用 group 融合，让它们并发推进并避免顺序造成的 deadlock。

```c
ncclGroupStart();
for (int peer = 0; peer < world_size; ++peer) {
  ncclSend(send_chunks[peer], send_counts[peer], dtype, peer, comm, stream);
  ncclRecv(recv_chunks[peer], recv_counts[peer], dtype, peer, comm, stream);
}
ncclGroupEnd();
```

这个形状可以表达 all-to-all、neighbor exchange 或 variable-size exchange。需要注意：
所有 rank 必须就 peer 顺序、count、datatype 和 buffer lifetime 达成一致。

### NCCL Stream Completion

NCCL collective 绑定到 CUDA stream。host 函数返回后，后续代码如果马上读取输出，需要
使用 CUDA stream/event 语义等待：

```c
ncclAllReduce(sendbuff, recvbuff, count, ncclFloat32, ncclSum, comm, comm_stream);

cudaEventRecord(done, comm_stream);
cudaStreamWaitEvent(user_stream, done, 0);

consume_recvbuff<<<grid, block, 0, user_stream>>>(recvbuff);
```

这个例子强调：通信完成不是单纯的 host-side return 问题，而是 stream ordering 问题。

## HCCL Examples

HCCL 是 Ascend 分布式训练中常见的 collective communication library。用户态训练框架
通常通过 `backend="hccl"` 走高层接口；做环境验证、连通性测试或基础性能测试时，
CANN 工具链里的 `hccl_test` 更直接。

`hccl_test` 适合用来回答三个问题：

- 多机、多卡环境是否能正常建链？
- 核心 collective 在当前 CANN、MPI、NPU 和网络配置下是否正确？
- 不同 message size 下的耗时和算法带宽大致是什么水平？

### HCCL Core Test Operations

HCCL 性能测试材料中推荐优先关注这几类操作：

| Operation | Test binary | Communication pattern | Typical use |
| ---- | ---- | ---- | ---- |
| AllReduce | `all_reduce_test` | many-to-many | 梯度聚合、参数同步。 |
| AllGather | `all_gather_test` | many-to-many | 数据聚合、参数收集。 |
| Broadcast | `broadcast_test` | one-to-many | 配置分发、初始化。 |
| AlltoAll | `alltoall_test` | many-to-many | 数据重排、负载均衡。 |

这四类基本覆盖了数据并行、参数/激活重组和 MoE-style reshuffle 的常见通信压力点。

### HCCL Environment Checklist

多机测试前，先检查 control plane 与硬件状态，而不是直接跑大流量测试：

| Check | Why it matters | Example |
| ---- | -------------- | ------- |
| SSH 免密 | MPI 需要拉起远端进程。 | `ssh root@<node_ip> "echo OK"` |
| CANN 版本一致 | 多机版本不一致容易导致建链或运行失败。 | 检查各节点 `version.cfg`。 |
| NPU 健康状态 | 故障卡会导致 collective 失败或 timeout。 | `npu-smi info -t health -i 0` |
| MPI 环境 | `hccl_test` 依赖 MPI 拉起多进程。 | MPICH 或 Open MPI。 |
| Hostfile | 定义节点和每个节点参与卡数。 | `175.99.1.3:8`。 |

一个双机 16 卡 hostfile 例子：

```text
175.99.1.3:8
175.99.1.4:8
```

### HCCL Tool Compilation

`hccl_test` 通常位于 Ascend toolkit 的 tools 目录。MPICH 环境下的编译形状如下：

```bash
export INSTALL_DIR=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpich/lib:${INSTALL_DIR}/lib64:$LD_LIBRARY_PATH

cd ${INSTALL_DIR}/tools/hccl_test
make MPI_HOME=/usr/local/mpich ASCEND_DIR=${INSTALL_DIR}
```

如果使用 Open MPI，则把 `MPI_HOME` 换成对应安装路径。

### HCCL Quick Connectivity Test

先用较小 message size 验证单机或多机连通性：

```bash
# 单机 8 卡
mpirun -n 8 ./bin/all_reduce_test -p 8 -b 8K -e 64M -f 2 -d fp32 -o sum

# 双机 16 卡
mpirun -f hostfile -n 16 ./bin/all_reduce_test -p 8 -b 8K -e 64M -f 2 -d fp32 -o sum
```

这里的参数可以这样读：

| Parameter | Meaning |
| --------- | ------- |
| `-n 8` / `-n 16` | MPI 启动的总进程数，也就是总 rank 数。 |
| `-p 8` | 单节点参与测试的 NPU 个数。 |
| `-b 8K` | 起始数据量。 |
| `-e 64M` | 结束数据量。 |
| `-f 2` | 每轮数据量乘法因子。 |
| `-d fp32` | datatype。 |
| `-o sum` | reduction operation。 |

这个测试对应本文前面的 DDP gradient all-reduce mental model：每个 rank 提供同类型、
同长度输入，HCCL 对所有 rank 的同位置元素执行 `sum`，再把结果提供给所有 rank。

### HCCL Full Bandwidth Test

连通性确认后，再把结束数据量拉到更接近训练大 bucket 的范围：

```bash
# 单机完整性能测试
mpirun -n 8 ./bin/all_reduce_test -p 8 -b 8K -e 1G -f 2 -d fp32 -o sum

# 多机完整性能测试
mpirun -f hostfile -n 16 ./bin/all_reduce_test -p 8 -b 8K -e 1G -f 2 -d fp32 -o sum
```

`-b 8K -e 1G -f 2` 会测试一串逐步翻倍的数据量，例如 8K、16K、32K，一直到 1G。
小消息主要暴露 latency 和 launch overhead，大消息更容易反映链路带宽与拓扑配置。

### HCCL all_gather Test

`all_gather_test` 可以验证 shard reconstruction 类通信：

```bash
./bin/all_gather_test -p 8 -b 8K -e 1G -f 2 -d fp32
```

如果把每个 rank 的输入看成一个 shard，完成后每个 rank 都应该得到按 rank order
聚合后的完整结果。这个 pattern 常用于参数分片或张量并行中的 activation/result
重组。

### HCCL broadcast Test

`broadcast_test` 可以验证 root-to-all 分发路径：

```bash
./bin/broadcast_test -p 8 -b 8K -e 1G -f 2 -d fp32
```

这个测试对应初始化参数、配置张量或 checkpoint metadata 的同步场景。调试时要特别注意
root rank 是逻辑 rank id，不是物理 NPU id；rank-to-device mapping 变化时，root 所在
物理卡也会变化。

### HCCL all_to_all Test

`alltoall_test` 用来验证 reshuffle 类通信：

```bash
./bin/alltoall_test -p 8 -b 8K -e 1G -f 2 -d fp32
```

这个测试更接近 MoE token routing 或 distributed layout transpose：每个 rank 都向
其他 rank 发送不同切片，同时接收发往自己的切片。调试这类问题时，除了带宽，还要关注
split size、destination order 和不均匀负载。

### HCCL Variable-Size Tests

`AlltoAllV`、`AllGatherV` 和 `ReduceScatterV` 这类 variable-size operation 更接近
真实 MoE 或稀疏路由场景：不同 rank 的 send/recv count 可以不同，因此参数配置比
fixed-size collective 更容易出错。

实操上建议按这个顺序排查：

1. 先确认 `all_reduce_test`、`all_gather_test`、`broadcast_test` 和 `alltoall_test`
   通过，排除基础建链和固定长度通信问题。
2. 再测试 variable-size operation，并保存每个 rank 的 split size 配置。
3. 如果出现 `retcode: 5` 这类变长参数错误，先检查 send/recv count 是否互相匹配，
   不要直接归因到硬件或网络。

### HCCL Result Reading

`hccl_test` 输出通常可以按这些字段阅读：

```text
data_size      avg_time(us)    alg_bandwidth(GB/s)    check_result
8192           125.3           0.065                  success
16384          132.1           0.124                  success
...
```

| Field | Meaning |
| ----- | ------- |
| `data_size` | 单个 NPU 上参与集合通信的数据量，通常按 bytes 理解。 |
| `avg_time(us)` | 该 message size 下的平均耗时。 |
| `alg_bandwidth(GB/s)` | 算法带宽，通常由数据量和耗时计算。 |
| `check_result` | 结果校验标识，例如 `success`、`failed` 或 `NULL`。 |

如果小消息慢但大消息正常，优先看 launch latency、rank placement 和测试参数。如果大消息
带宽异常，优先看链路拓扑、网卡选择、跨节点配置、CANN/MPI 版本和是否有故障卡或残留进程。

## Basic Data-Parallel Communication Flow

最简单的分布式训练心智模型是同步数据并行：

1. 每个 rank 持有同一个模型的副本。
2. 每个 rank 接收不同的输入 batch shard。
3. 前向和反向计算本地梯度。
4. process group 对梯度 bucket 执行 `all_reduce`。
5. 每个 rank 按需对 reduce 后的梯度做除法或其他归一化。
6. 每个 rank 应用相同的优化器更新，并保持模型副本同步。

初始化阶段也可能使用通信。常见模式是让 rank 0 持有初始参数值，然后通过 `broadcast`
分发，使每个 rank 都从相同模型状态开始。checkpoint 加载、随机种子设置、metric 聚合
和分布式验证也可能在主训练 step 周围加入更多 collective。

对于基础数据并行任务，高层通信预算大致如下：

| Operation | Payload | When it happens | Frequency | Approximate payload size |
| ---- | ---- | -------- | ---- | ------------ |
| `broadcast` | 模型参数和 buffer | 某个 rank 初始化或加载状态之后 | 通常每次启动或加载一次 | 模型参数和 buffer 大小 |
| `all_reduce` | 梯度 | 反向传播产生梯度 bucket 之后 | 每个同步训练 step 都会发生，通常每个 bucket 一次 | group 内 `world_size *` 梯度 bucket 大小 |

这就是为什么当模型已经能放进单个设备时，数据并行既简单又有效：主要的重复通信是梯度同步。
它本身不会降低保存一个完整模型副本所需的内存。更大的模型通常需要张量并行、流水线并行、
FSDP/ZeRO 风格切分，或这些方式的组合。

backend 因此必须同时保证值正确性和顺序正确性。值正确性很直观：正确的字节需要到达正确
的 rank。顺序正确性同样重要：如果 rank 0 调用 `all_reduce` 而 rank 1 调用
`broadcast`，两个 rank 可能永久等待，或报告 backend 错误。

## Control Plane And Data Plane

把 control plane 和 data plane 分开看，会更容易理解 distributed code。

### Control Plane

Control plane 移动少量 metadata：

- 哪个进程是 rank 0？
- `world_size` 是多少？
- 哪台 host 和哪个 port 拥有 rendezvous 服务？
- 各 rank 应使用哪个 communicator bootstrap token？
- 某个 key 是否已经被设置，使另一个 rank 可以继续执行？
- hostfile 中每台机器应该拉起几个 rank？

常见 control plane 组件包括：

- `torchrun`、MPI、SLURM 或其他 launcher。
- PyTorch Store、MPI broadcast、socket 或文件系统 rendezvous。
- NCCL 的 `ncclUniqueId` 分发。
- HCCL 测试中的 hostfile、MPI 环境和 toolkit 路径。

Store、hostfile 和 unique id 不是 model tensor。它们是小型 metadata，用于让所有 rank
在 device communication 开始前达成一致。

### Data Plane

Data plane 在 device interconnect 或网络上移动 tensor payload。它通常包括：

- communicator handle，例如 `ncclComm_t` 或 HCCL communicator。
- backend collective API，例如 `ncclAllReduce`、`HcclAllReduce`。
- 设备 stream 和 event。
- datatype、reduction op、root rank、split size 等 payload metadata。
- 异步错误查询、abort/destroy/finalize 等 lifecycle API。

control plane 出错时，常见症状是 rank 启动失败、无法 rendezvous、hostfile 不匹配、
unique id 没有分发到所有进程。data plane 出错时，常见症状是 collective hang、带宽异常、
结果校验失败、stream 同步错误或设备侧 runtime error。

## PyTorch c10d 如何参与

PyTorch 的 distributed 包在 C++ 实现中通常称为 c10d。重要抽象层如下：

```text
Python torch.distributed API
  -> torch._C._distributed_c10d bindings
  -> c10d::ProcessGroup / c10d::Backend
  -> backend-specific implementation
  -> communication library and device runtime
```

从用户视角看，最常见的入口是：

```python
dist.init_process_group(backend="nccl")
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
work = dist.all_gather_into_tensor(output, input, async_op=True)
work.wait()
```

`async_op=True` 返回的 `Work` handle 并不改变 collective 的参与规则：所有 rank 仍然
必须以相同顺序进入兼容的 operation。它只是让 host 线程可以在等待前做其他工作。

## Stream, Event, And Memory Lifetime Concepts

集合通信不只是一次函数调用。它通常是提交到 stream 上的设备任务。

在设备 backend 中，通信 stream 必须和用户计算 stream 建立正确依赖。在通信操作读取输入
张量之前，通信 stream 必须等待该张量当前 stream 上更早的任务。通信写入输出张量之后，
后续用户任务必须等待通信 stream。

event 是这种交接的常见排序机制。

内存生命周期也很重要。如果一个异步通信 stream 还没完成读写，而张量已经离开作用域，
allocator 可能过早复用这块内存。框架通常会通过暂存张量、记录 stream 或延迟释放来保护
参与张量，直到 work 完成。

这就是为什么初学者在调试异步 collective 失败前，应先理解 stream/event 语义。

## DDP And Gradient Buckets

DistributedDataParallel 不会对每个参数分别执行 all-reduce。它会把梯度分组成 bucket，
然后对这些 bucket 执行通信。

原因是性能。一个大模型可能有成千上万个 parameter；如果每个 parameter ready 后都单独
发起一次 collective，通信调用数量会非常多，latency 成本会压过 bandwidth。DDP 的
`Reducer` 会按 parameter 顺序和 bucket size 把多个 gradient 放到同一个 bucket。
backward 过程中，当某个 bucket 内所有 gradient 都 ready 后，DDP 就可以对这个 bucket
发起 `all_reduce`，同时 autograd 继续计算后面层的 gradient。

从时间线上看，DDP 希望实现 compute/communication overlap：

```text
backward computes later-layer gradients
  -> bucket A becomes ready
  -> all_reduce(bucket A) starts on communication stream
  -> backward continues computing earlier-layer gradients
  -> bucket B becomes ready
  -> all_reduce(bucket B) starts
  -> wait for outstanding bucket work before optimizer.step()
```

因此，`loss.backward()` 返回前，DDP 通常已经安排并等待了必要的 bucket communication。
返回之后，每个 rank 的 `parameter.grad` 应该已经是同步后的 gradient，`optimizer.step()`
才能在所有 replica 上做一致更新。

关键概念是：DDP 是 collective 的更高层用户。如果 `all_reduce` 单独可用但 DDP 失败，
bug 可能在 bucket 顺序、参数一致性、autograd 时序、stream 同步或 reducer 集成中，而
不一定在原始 backend collective 调用里。

## Rendezvous And Launch

collective 能运行之前，各 rank 需要一个共同的会合点。

常见启动模型包括：

- `torchrun --nproc_per_node=8 ...`：适合 PyTorch 原生分布式作业。
- `mpirun -n 8 ...`：适合 NCCL/HCCL test 或 MPI 风格程序。
- `mpirun -f hostfile -n 16 ...`：适合多机测试，hostfile 描述节点与每节点卡数。
- 集群调度系统注入 `RANK`、`WORLD_SIZE`、`LOCAL_RANK`、master address 等环境变量。

重要的心智模型是：

```text
launcher decides process environment
  -> rendezvous decides rank and world size
  -> control plane exchanges communicator metadata
  -> process group performs tensor communication
```

## Debugging And Failure Concepts

collective 对顺序敏感。process group 中的每个 rank 都必须以相同的 collective 序列、
兼容的张量元数据进行调用。任何不匹配都可能导致 hang、timeout、结果错误或 backend error。

调试时优先检查这些问题：

- 是否所有 rank 都进入了同一个 operation？
- `world_size`、rank id、local rank 和 device mapping 是否一致？
- root rank 是 rank id，不是物理 device id，是否被混淆？
- tensor shape、dtype、count 和 split size 是否一致？
- `all_to_allv` 或 variable-size P2P 的 send/recv count 是否互相匹配？
- 是否有某个 rank 提前异常退出，导致其他 rank 在 collective 中等待？
- stream 上的生产者任务是否已经被通信 stream 等待？
- 输出张量被后续 compute stream 使用前，是否已经等待通信完成？
- 多机测试时 SSH、hostfile、CANN/CUDA toolkit、driver、MPI 和网卡选择是否一致？

对于 NCCL，常用排查方向包括 `NCCL_DEBUG`、网络接口选择、InfiniBand/RoCE 配置、拓扑识别
和 async error。对于 HCCL，先确认 `npu-smi` 健康状态、CANN 版本一致、MPI 能正确拉起
所有 rank、hostfile 没有跨错误节点组合，并查看 Ascend 运行日志。

## Concepts To Learn Next

要深入理解集合通信，可以按顺序学习以下主题。

### 1. PyTorch Distributed c10d

重点关注：

- `init_process_group`
- `ProcessGroup`
- `Store`
- `Work`
- `ReduceOp`
- `DistributedDataParallel`
- rendezvous backend

这解释了高层 API 如何把 collective 请求交给具体 backend。

### 2. Collective Algorithms And Tensor Shapes

重点关注：

- all-reduce 与 reduce-scatter 的区别
- all-gather 与 all-to-all 的区别
- broadcast 和 reduce 的 root rank 语义
- 非均匀 split size
- dtype 与 layout 兼容性
- 为什么所有 rank 必须以相同顺序调用操作

这解释了 backend 为什么要检查张量元数据和 sequence number。

### 3. Process Launch And Networking

重点关注：

- rank 和 local rank
- world size 和 node rank
- host 和 port 选择
- TCPStore 或 MPI 风格 bootstrap
- hostfile
- timeouts

这解释了为什么很多 collective hang 实际是启动、rendezvous 或网络配置问题。

### 4. Device Runtime Stream And Event

重点关注：

- current device
- current stream
- communication stream
- event record 和 wait
- stream synchronization
- allocator stream recording

这解释了为什么 collective backend 必须协调 stream，并在 Python 调用返回后继续保护张量
生命周期。

### 5. Communication Library Runtime

重点关注：

- communicator 创建
- bootstrap token 交换
- communicator 缓存
- async error 查询
- group start 和 group end
- library feature probing

这解释了 NCCL、HCCL 等 backend 的公共结构。

### 6. Distributed Training Integration

重点关注：

- autograd 梯度产生顺序
- DDP reducer bucket
- communication hook
- 梯度缩放和平均
- 参数一致性检查

这解释了为什么训练框架除了原始 collective 方法之外，还需要 reducer、hook、bucket 和
错误传播机制。

## External References

- [NVIDIA NCCL repository](https://github.com/NVIDIA/nccl)：用于理解标准 GPU collective
  词汇、NCCL 支持的通信例程以及 `nccl-tests` 的入口。
- [NCCL communicator creation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)：用于理解
  `ncclUniqueId` bootstrap、rank 到 CUDA device 的绑定和 `ncclCommInitRank`。
- [NCCL collective operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)：用于理解
  NCCL collective API name、payload semantics、root rank 和 rank-order layout。
- [NCCL point-to-point communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html)：用于理解
  `ncclSend` / `ncclRecv`、grouped P2P、scatter/gather/all-to-all-style pattern。
- [NCCL CUDA stream semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html)：用于理解为什么必须通过
  device stream 和 event 推理通信完成状态。
- [NCCL group calls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html)：用于理解一个线程管理多 GPU、
  聚合多个 collective 和 P2P group 的语义。
- [Ascend HCCL performance test skill](https://github.com/ascend-ai-coding/awesome-ascend-skills/blob/main/skills/training/hccl-test/SKILL.md)：用于
  HCCL `hccl_test` 的核心算子、编译、运行参数、hostfile 和结果解析示例。

## 继续阅读

- LLM Tensor Parallelism Patterns
- PyTorch Training Support Scope
