import asyncio
import torch
import uuid
import time

from pipeline_parallel import get_pipeline_parallel_flow
from matplotlib import pyplot as plt

task_timings = []
global_start = time.time()

# returns a list of integers close to a list of floats, credit to ChatGPT
def round_with_sum_constraint(floats, target_sum=None):
    if target_sum is None:
        target_sum = round(sum(floats))

    # First, floor all the numbers
    floored = [int(x) for x in floats]
    remainder = target_sum - sum(floored)

    # Get the fractional parts, sorted by who has the largest decimal
    fractional_parts = sorted(
        [(i, floats[i] - floored[i]) for i in range(len(floats))],
        key=lambda x: x[1],
        reverse=True
    )

    # Distribute the remaining +1s to the highest fractional parts
    for i in range(remainder):
        idx = fractional_parts[i][0]
        floored[idx] += 1

    return floored

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

# related to worker.py
class MockWorker():

    def __init__(self, training_rate):
        self.training_rate = training_rate
        self.num_blocks = 10

    async def run_net(self, x, URL, direction, call_id, return_outputs=False, clear_local_cache=False, clear_remote_cache=False, save_tensors=False):
        await asyncio.sleep(0.001*self.num_blocks/self.training_rate)
            
    def set_blocks(self, num_blocks):
        self.num_blocks = num_blocks

def offload(source, sink):
    source.set_blocks(source.num_blocks-1)
    sink.set_blocks(sink.num_blocks+1)

async def main():

    num_workers = 3
    total_blocks = 8

    worker_training_rates = [torch.randint(8, (1, )).item()/5+0.2 for _ in range(num_workers)]
    stubs = [MockWorker(rate) for rate in worker_training_rates]

    # each worker is allocated a number of blocks proportional to its training rate
    total_training_rate = sum(stub.training_rate for stub in stubs)
    allocations = [total_blocks*stub.training_rate/total_training_rate for stub in stubs]
    allocations = round_with_sum_constraint(allocations, target_sum=total_blocks)
    for i in range(num_workers): stubs[i].set_blocks(allocations[i])

    for batch_index in range(1):

        batch = torch.randn([10, 28, 28])

        def get_pipeline_stages(j, x):
            pipeline_stages = []
            ctx_id = uuid.uuid4()

            pipeline_stages.append(metrics_wrapper(f"f1s{j+1}", stubs[0].run_net(x, "URL", 'forward', ctx_id, save_tensors=True)))
            pipeline_stages.append(metrics_wrapper(f"f2s{j+1}", stubs[1].run_net(x, "URL", 'forward', ctx_id, save_tensors=True)))
            pipeline_stages.append(metrics_wrapper(f"f3s{j+1}", stubs[2].run_net(x, "URL", 'forward', ctx_id, save_tensors=True, return_outputs=True)))

            pipeline_stages.append(metrics_wrapper(f"b3s{j+1}", stubs[2].run_net(x, "URL", 'backward', ctx_id, clear_remote_cache=True)))
            pipeline_stages.append(metrics_wrapper(f"b2s{j+1}", stubs[1].run_net(x, "URL", 'backward', ctx_id, clear_remote_cache=True)))
            pipeline_stages.append(metrics_wrapper(f"b1s{j+1}", stubs[0].run_net(x, "URL", 'backward', ctx_id, clear_remote_cache=True, return_outputs=True, clear_local_cache=True)))

            return pipeline_stages

        flow = get_pipeline_parallel_flow(3, get_pipeline_stages, batch)

        await flow.start()

        plot_allocation_timings()



asyncio.run(main())