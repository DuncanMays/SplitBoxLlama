import asyncio
import torch
import uuid
import time

from pipeline_parallel import get_pipeline_parallel_flow
from matplotlib import pyplot as plt

task_timings = []
global_start = time.time()

def plot_allocation_timings():
    print("plotting!")

    f_x = []
    f_tops = []
    f_bottoms = []
    b_x = []
    b_tops = []
    b_bottoms = []

    for dp in task_timings:
        d, w = dp["task_str"][0], dp["task_str"][1]

        if (d=="f"):
            f_x.append(w)
            f_tops.append(dp["end"] - dp["start"])
            f_bottoms.append(dp["start"])

        if (d=="b"):
            b_x.append(w)
            b_tops.append(dp["end"] - dp["start"])
            b_bottoms.append(dp["start"])

    plt.bar(f_x, f_tops, bottom=f_bottoms, edgecolor="black", color="red")
    plt.bar(b_x, b_tops, bottom=b_bottoms, edgecolor="black", color="blue")

    plt.show()

def metrics_wrapper(task_str, coro):

    async def wrapped():
        start = time.time() - global_start

        # print(f'starting {task_str}')
        result = await coro
        # print(f'{task_str} ended')

        end = time.time() - global_start

        data_point = {"task_str": task_str, "start": start, "end": end}
        task_timings.append(data_point)

        return result

    return wrapped()

# from worker.py
class MockNeuralBlock():

    def __init__(self, delay_factor):
        self.delay_factor = delay_factor
        self.num_blocks = 0.01

    async def run_net(self, x, URL, direction, call_id, return_outputs=False, clear_local_cache=False, clear_remote_cache=False, save_tensors=False):
        await asyncio.sleep(self.num_blocks*self.delay_factor)
            
    def get_outputs(self, call_id, clear_cache=False):
        pass

async def main():

    batch = torch.randn([15, 28, 28])

    # the worker can train one batch of data on one block at this rate, let's say 5
    # so the worker can train 5 blocks at one batch per second
    # or it could train one block at 5 batches per second
    # if n is the number of batches, and m is the number of blocks, s is the number of seconds
    # n*m = 5*s
    block_batch_per_second = 5

    stub_1 = MockNeuralBlock(torch.randint(100, (1, )).item()/30)
    stub_2 = MockNeuralBlock(torch.randint(100, (1, )).item()/30)
    stub_3 = MockNeuralBlock(torch.randint(100, (1, )).item()/30)

    def get_pipeline_stages(j, x):
        pipeline_stages = []
        ctx_id = uuid.uuid4()

        pipeline_stages.append(metrics_wrapper(f"f1s{j+1}", stub_1.run_net(x, "URL", 'forward', ctx_id, save_tensors=True)))
        pipeline_stages.append(metrics_wrapper(f"f2s{j+1}", stub_2.run_net(x, "URL", 'forward', ctx_id, save_tensors=True)))
        pipeline_stages.append(metrics_wrapper(f"f3s{j+1}", stub_3.run_net(x, "URL", 'forward', ctx_id, save_tensors=True, return_outputs=True)))

        pipeline_stages.append(metrics_wrapper(f"b3s{j+1}", stub_3.run_net(x, "URL", 'backward', ctx_id, clear_remote_cache=True)))
        pipeline_stages.append(metrics_wrapper(f"b2s{j+1}", stub_2.run_net(x, "URL", 'backward', ctx_id, clear_remote_cache=True)))
        pipeline_stages.append(metrics_wrapper(f"b1s{j+1}", stub_1.run_net(x, "URL", 'backward', ctx_id, clear_remote_cache=True, return_outputs=True, clear_local_cache=True)))

        return pipeline_stages

    flow = get_pipeline_parallel_flow(3, get_pipeline_stages, batch)

    await flow.start()

    plot_allocation_timings()

asyncio.run(main())