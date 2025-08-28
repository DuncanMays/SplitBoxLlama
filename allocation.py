import asyncio
import torch
import uuid
import time

from pipeline_parallel import get_pipeline_parallel_flow, parse_task_str
from plot_pipeline import metrics_wrapper, plot_timings
from benchmark import benchmark, MockWorker

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

async def main():

    num_workers = 3
    total_blocks = 20
    num_mini_batches = 10

    worker_compute_rates = [torch.randint(15, (1, )).item()/15+0.1 for _ in range(num_workers)]
    worker_download_rates = [torch.randint(200, (1, )).item()+50 for _ in range(num_workers)]
    stubs = [MockWorker(forward_rate=c, backward_rate=1.2*c, download_rate=d) for c, d in zip(worker_compute_rates, worker_download_rates)]

    # benchmark workers
    benchmark_promises = [benchmark(stub, torch.randn([32, 28, 28]), 20) for stub in stubs]
    benchmark_scores = await asyncio.gather(*benchmark_promises)

    C = [score[0] for score in benchmark_scores]
    D = [score[1] for score in benchmark_scores]

    # each worker is allocated a number of blocks so that the expected delays of all workers are the same
    expected_delays = [total_blocks*c + d for c, d in zip(C, D)]
    t = [round(1/t) for t in expected_delays]

    print(expected_delays)
    print(t)
    allocations = round_with_sum_constraint(t, target_sum=total_blocks)
    print(allocations)
    for i in range(num_workers): stubs[i].set_blocks(allocations[i])

    exit()

    for batch_index in range(1):

        batch = torch.randn([num_mini_batches, 28, 28])

        def get_pipeline_stages(j, x):
            pipeline_stages = []
            ctx_id = uuid.uuid4()

            # for i in range(num_workers-1):
            #     pipeline_stages.append(metrics_wrapper(f"f{i+1}s{j+1}", stubs[i].run_net(x, "URL", 'forward', ctx_id, save_tensors=True)))

            # pipeline_stages.append(metrics_wrapper(f"f{len(stubs)}s{j+1}", stubs[-1].run_net(x, "URL", 'forward', ctx_id, save_tensors=True, return_outputs=True)))

            # for i in range(num_workers-1, 0, -1):
            #     pipeline_stages.append(metrics_wrapper(f"b{i+1}s{j+1}", stubs[i].run_net(x, "URL", 'backward', ctx_id, clear_remote_cache=True)))

            # pipeline_stages.append(metrics_wrapper(f"b1s{j+1}", stubs[0].run_net(x, "URL", 'forward', ctx_id, save_tensors=True, return_outputs=True)))

            # if the context already has an ID, that means it's been through a FnStub already
            # this could be a problem for recursive patterns, in that case, the stub's context store will need to be a dict of lists of contexts
            if not hasattr(ctx, 'id'):
                ctx.id = uuid.uuid4()

            stubs[0].load_activations(ctx.id, x)

            for i in range(len(stubs)):
                if (i != 0): stubs[i].fetch_activations(ctx.id, 'url')
                stubs[i].forward(ctx.id)

            x = stubs[-1].get_activations(ctx.id)

            # return x

            stub_3.load_gradients(ctx.id, g)
            stub_3.backward(ctx.id, clear_cache=True)

            stub_2.fetch_gradients(ctx.id, url_3, clear_cache=True)
            stub_2.backward(ctx.id, clear_cache=True)

            stub_1.fetch_gradients(ctx.id, url_2, clear_cache=True)
            stub_1.backward(ctx.id, clear_cache=True)
            x = stub_1.get_gradients(ctx.id, clear_cache=True)

            # return g

            # return pipeline_stages

        flow = get_pipeline_parallel_flow(num_workers, get_pipeline_stages, batch)

        print('executing flow')
        await flow.start()

        print("plotting!")
        plot_timings()

if (__name__ == "__main__"): asyncio.run(main())