import axon
import asyncio
import torch
import time
import uuid
import cloudpickle
import pickle

from SplitBox.benchmark import benchmark
from SplitBox.allocation import allocate, round_with_sum_constraint, delay
from SplitBox.pipeline_parallel import get_pipeline_parallel_flow, get_pipeline_forward_flow
from SplitBox.multi_stub import get_multi_stub
from SplitBox.worker import NeuralBlock
from SplitBox.tracer import TID_COMPUTE, TID_COMM

async def benchmark(worker, x):
    start = time.time()

    await worker.load_activations('activation_id', x)

    net_time = time.time() - start
    start = time.time()

    await worker.forward('activation_id')

    comp_time = time.time() - start
    return comp_time, net_time

def get_training_flow(stubs, urls, batch, target, criterion, client_tracer=None):

    losses = []
    criterion_str = cloudpickle.dumps(criterion)

    def get_pipeline_stages(pipeline_num, x, y):
        pipeline_stages = []
        ctx_id = uuid.uuid4()

        for worker_num in range(len(stubs)):

            # This function doesn't execute while this loop is iterating, it does when the returned event flow is started and awaited
            # The value of worker_num will have changed when the function runs, so it needs to capture the value of worker_num in a new variable, i
            # That's why we have the parameter i
            async def forward_stage(i=worker_num):

                # first stage loads activations, otherwise fetch from last worker
                if (i == 0): await stubs[i].load_activations(ctx_id, x.clone())
                else: await stubs[i].fetch_activations(ctx_id, urls[i-1])
                
                if (i == len(stubs)-1):
                    # last stage is sent training targets and returns loss
                    loss = await stubs[i].final_stage(ctx_id, y.clone(), criterion_str, loss_scale=1.0/len(batch))
                    losses.append(loss)
                else:
                    await stubs[i].forward(ctx_id)

            stage_id = f"f{worker_num+1}s{pipeline_num+1}"
            coro = forward_stage()
            if client_tracer is not None: coro = client_tracer.wrap(stage_id, coro, pid=worker_num)
            pipeline_stages.append(coro)

        for worker_num in range(len(stubs)-1, -1, -1):

            async def backward_stage(i=worker_num):

                if (i != len(stubs)-1): await stubs[i].fetch_gradients(ctx_id, urls[i+1], clear_cache=True)
                await stubs[i].backward(ctx_id, clear_cache=True)

            stage_id = f"b{worker_num+1}s{pipeline_num+1}"
            coro = backward_stage()
            if client_tracer is not None: coro = client_tracer.wrap(stage_id, coro, pid=worker_num)
            pipeline_stages.append(coro)

        return pipeline_stages

    flow = get_pipeline_parallel_flow(len(stubs), get_pipeline_stages, batch, target)

    return flow, losses

def get_eval_flow(stubs, urls, batch, client_tracer=None):

    outputs = []

    def get_pipeline_stages(pipeline_num, x, _y):
        pipeline_stages = []
        ctx_id = uuid.uuid4()

        for worker_num in range(len(stubs)):

            async def forward_stage(i=worker_num):

                if i == 0: await stubs[i].load_activations(ctx_id, x.clone())
                else: await stubs[i].fetch_activations(ctx_id, urls[i-1])

                await stubs[i].forward(ctx_id, inference=True)

                if i == len(stubs) - 1:
                    y = await stubs[i].get_activations(ctx_id, clear_cache=True)
                    outputs.append(y)

            stage_id = f"f{worker_num+1}s{pipeline_num+1}"
            coro = forward_stage()
            if client_tracer is not None: coro = client_tracer.wrap(stage_id, coro, pid=worker_num)
            pipeline_stages.append(coro)

        return pipeline_stages

    dummy_target = [None] * len(batch)
    flow = get_pipeline_forward_flow(len(stubs), get_pipeline_stages, batch, dummy_target)

    return flow, outputs