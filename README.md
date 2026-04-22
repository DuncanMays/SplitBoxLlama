# SplitBox

SplitBox is a Python library for distributed pipeline-parallel training of PyTorch neural networks. It splits a model into sequential stages, places each stage on a separate worker, and coordinates forward and backward passes across the cluster using an event-driven async scheduler.

## How It Works

A model is divided into a sequence of **neural blocks** — each block wraps a `nn.Module` together with its optimizer and scheduler. Blocks are assigned to **workers** (remote processes), which communicate via the [Axon](https://github.com/DuncanMays/axon-ECRG) RPC framework. During training, SplitBox pipelines mini-batches through the stages: while one worker runs the backward pass for batch *n*, the next worker is already running the forward pass for batch *n+1*.

### Core Components

| Module | Description |
|---|---|
| `worker.py` | `NeuralBlock`, `BlockStack`, and `Worker` — the stateful compute unit on each machine |
| `pipeline_parallel.py` | Builds the dependency graph (EventFlow) that orders forward and backward tasks across all workers |
| `pipeline_client.py` | High-level `get_training_flow()` and `get_eval_flow()` for driving a full training or inference run |
| `events/EventFlow.py` | Async task-orchestration engine: schedules callbacks when named events fire, handles concurrency with asyncio |
| `multi_stub.py` | Fan-out RPC facade — one method call dispatched to all workers simultaneously |
| `allocation.py` | Linear-programming solver (OR-Tools) that balances block load across heterogeneous workers |
| `benchmark.py` | Measures per-worker compute rate and network latency to feed the allocator |
| `plot_pipeline.py` | Visualises the pipeline execution timeline as a stacked bar chart |

### Pipeline Schedule

SplitBox uses a 1F1B (one-forward-one-backward) schedule. The scheduler is built by `get_pipeline_parallel_flow()`, which names every task `{direction}{worker}s{stage}` (e.g. `f1s1` = forward, worker 1, stage 1) and wires their dependencies so that:

1. Each forward stage waits for the previous worker's forward stage to finish.
2. Each backward stage waits for the next worker's backward stage and for its own forward stage to complete.

## Experiments

The repository includes several experiments on image classification benchmarks.

### VGG16 on CIFAR-10 (`/VGG`)

Trains a VGG-16 network on CIFAR-10 using a 3-stage distributed pipeline. The network is split into three `BlockStack`s in `VGG_blocks.py`; `VGG_pipeline.py` spins up workers and orchestrates training with `get_training_flow()`.

### ResNet18 on CIFAR-10 (`/ResNet_CIFAR10`)

Provides a local single-machine baseline (`local_train.py`) and a 3-stage pipeline split (`ResNetStages.py`) for ResNet-18 on CIFAR-10. The baseline is useful for comparing accuracy and throughput against the distributed version.

## Installation

```bash
pip install -r pip_env.txt
pip install axon-ECRG
```

## Project Layout

```
SplitBox/               # Library source
├── events/             # EventEmitter and EventFlow
├── tests/              # Unit and integration tests
VGG/                    # VGG16 CIFAR-10 experiment
ResNet_CIFAR10/         # ResNet18 CIFAR-10 experiment
```
