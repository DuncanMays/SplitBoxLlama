import asyncio
import torch
import uuid
import time

from pipeline_parallel import get_pipeline_parallel_flow, parse_task_str
from plot_pipeline import metrics_wrapper, plot_timings

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

# related to worker.py
class MockWorker():

    def __init__(self, training_rate):
        self.training_rate = training_rate
        self.num_blocks = 10

    async def run_net(self, x, URL, direction, call_id, return_outputs=False, clear_local_cache=False, clear_remote_cache=False, save_tensors=False):
        await asyncio.sleep(0.001*self.num_blocks/self.training_rate)
            
    def set_blocks(self, num_blocks):
        self.num_blocks = num_blocks

async def main():

    num_workers = 5
    total_blocks = 20
    num_mini_batches = 10

    worker_training_rates = [torch.randint(8, (1, )).item()/5+0.2 for _ in range(num_workers)]
    # worker_training_rates = [1 for _ in range(num_workers)]
    stubs = [MockWorker(rate) for rate in worker_training_rates]

    # each worker is allocated a number of blocks proportional to its training rate
    total_training_rate = sum(stub.training_rate for stub in stubs)
    allocations = [total_blocks*stub.training_rate/total_training_rate for stub in stubs]
    allocations = round_with_sum_constraint(allocations, target_sum=total_blocks)
    print(allocations)
    for i in range(num_workers): stubs[i].set_blocks(allocations[i])

    for batch_index in range(1):

        batch = torch.randn([num_mini_batches, 28, 28])

        def get_pipeline_stages(j, x):
            pipeline_stages = []
            ctx_id = uuid.uuid4()

            for i in range(num_workers-1):
                pipeline_stages.append(metrics_wrapper(f"f{i+1}s{j+1}", stubs[i].run_net(x, "URL", 'forward', ctx_id, save_tensors=True)))

            pipeline_stages.append(metrics_wrapper(f"f{len(stubs)}s{j+1}", stubs[-1].run_net(x, "URL", 'forward', ctx_id, save_tensors=True, return_outputs=True)))

            for i in range(num_workers-1, 0, -1):
                pipeline_stages.append(metrics_wrapper(f"b{i+1}s{j+1}", stubs[i].run_net(x, "URL", 'backward', ctx_id, clear_remote_cache=True)))

            pipeline_stages.append(metrics_wrapper(f"b1s{j+1}", stubs[0].run_net(x, "URL", 'forward', ctx_id, save_tensors=True, return_outputs=True)))


            return pipeline_stages

        flow = get_pipeline_parallel_flow(num_workers, get_pipeline_stages, batch)

        print('executing flow')
        await flow.start()

        print("plotting!")
        plot_timings()



asyncio.run(main())