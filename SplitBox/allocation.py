import asyncio
import torch
import uuid
import time

from ortools.linear_solver import pywraplp

from SplitBox.pipeline_parallel import get_pipeline_parallel_flow, parse_task_str
from SplitBox.plot_pipeline import metrics_wrapper, plot_timings
from SplitBox.benchmark import benchmark, MockWorker

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

def delay(b, c, d):
    if b == 0:
        return b*c
    else:
        return b*c + d

def allocate(benchmark_scores, num_blocks):

    num_scores = len(benchmark_scores)
    solver = pywraplp.Solver.CreateSolver('GLOP')  # 'GLOP' is for linear programming

    # variables
    block_allocations = []
    for i in range(num_scores):
        block_allocations.append(solver.NumVar(0, solver.infinity(), f'b{i}'))

    max_delay = solver.NumVar(0, solver.infinity(), f'max_delay')

    # constraints
    solver.Add(sum(block_allocations) == num_blocks)
    
    for i in range(num_scores):
        b = block_allocations[i]
        c = benchmark_scores[i][0]
        d = benchmark_scores[i][1]

        solver.Add(delay(b, c, d) <= max_delay)

    # objective function
    solver.Minimize(max_delay)

    # Solve the problem
    status = solver.Solve()

    # Output results
    if status == pywraplp.Solver.OPTIMAL:
        print(f'Objective value = {solver.Objective().Value()}')
    else:
        print('The problem does not have an optimal solution.')

    return [b.solution_value() for b in block_allocations]

async def main():

    num_workers = 3
    total_blocks = 20
    num_mini_batches = 10

    worker_compute_rates = [torch.randint(15, (1, )).item()/15+0.1 for _ in range(num_workers)]
    worker_download_rates = [torch.randint(200, (1, )).item()+50 for _ in range(num_workers)]
    stubs = [MockWorker(forward_rate=c, backward_rate=1.2*c, download_rate=d) for c, d in zip(worker_compute_rates, worker_download_rates)]

    # benchmark workers
    print('benchmarking!')
    benchmark_promises = [benchmark(stub, torch.randn([32, 28, 28]), 20) for stub in stubs]
    benchmark_scores = await asyncio.gather(*benchmark_promises)

    C = [score[0] for score in benchmark_scores]
    D = [score[1] for score in benchmark_scores]
    print("benchmark scores: ", C, D)

    # each worker is allocated a number of blocks so that the expected delays of all workers are the same
    relative_delays = [total_blocks*c + d for c, d in zip(C, D)]
    print("relative delays: ", relative_delays)

    allocations = allocate(benchmark_scores, total_blocks)
    print('allocations:', allocations)
    allocations = round_with_sum_constraint(allocations, target_sum=total_blocks)
    print('rounded allocations:', allocations)

    expected_delays = [delay(b, c, d) for b, c, d in zip(allocations, C, D)]
    print("expected delays: ", expected_delays)

    print('====================================================')

    # stubs[0] = MockWorker(forward_rate=0.1, backward_rate=0.1, download_rate=150)

    for batch_index in range(1):

        batch = torch.randn([num_mini_batches, 28, 28])

        def get_pipeline_stages(j, x):
            pipeline_stages = []
            ctx_id = uuid.uuid4()

            for _i in range(len(stubs)):

                async def next_stage(i=_i):
                    if (i == 0): await stubs[i].load_activations(ctx_id, x)
                    if (i != 0): await stubs[i].fetch_activations(ctx_id, 'url')
                    await stubs[i].forward(ctx_id)

                pipeline_stages.append(metrics_wrapper(f"f{_i+1}s{j+1}", next_stage()))

            # calculate loss

            for _i in range(len(stubs)-1, -1, -1):

                async def next_stage(i=_i):
                    if (i != len(stubs)-1): await stubs[i].fetch_gradients(ctx_id, 'url', clear_cache=True)
                    await stubs[i].backward(ctx_id, clear_cache=True)

                pipeline_stages.append(metrics_wrapper(f"b{_i+1}s{j+1}", next_stage()))

            return pipeline_stages

        flow = get_pipeline_parallel_flow(num_workers, get_pipeline_stages, batch)

        print('executing flow')
        await flow.start()

        print("plotting!")
        plot_timings()

if (__name__ == "__main__"): asyncio.run(main())