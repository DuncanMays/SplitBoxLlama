import axon
import asyncio
import torch
import time
import uuid
import cloudpickle
import pickle

from benchmark import benchmark
from allocation import allocate, round_with_sum_constraint, delay
from pipeline_parallel import get_pipeline_parallel_flow
from plot_pipeline import metrics_wrapper, plot_timings
from multi_stub import get_multi_stub
from worker import NeuralBlock
from llama_blocks import LlamaBlock
from config import MASTER_CONFIG

url_1 = "192.168.2.19:8001/llama_worker"
url_2 = "192.168.2.19:8002/llama_worker"
url_3 = "192.168.2.19:8003/llama_worker"
# url_3 = "192.168.2.44:8001/llama_worker"

async def benchmark(worker, x):
    start = time.time()

    await worker.load_activations('activation_id', x)

    net_time = time.time() - start
    start = time.time()

    await worker.forward('activation_id')

    comp_time = time.time() - start
    return comp_time, net_time

async def main():
    total_blocks = 20
    num_mini_batches = 10

    criterion = torch.nn.functional.cross_entropy
    criterion_str = cloudpickle.dumps(criterion)

    urls = [url_1, url_2, url_3]
    stubs = [axon.client.get_stub(url) for url in urls]

    block_stubs = [axon.client.get_stub(url+"/net") for url in urls]
    multi_block_stub = get_multi_stub(block_stubs)

    # benchmark workers
    print('setting up!')

    num_workers = len(stubs)
    benchmark_block = NeuralBlock(lambda : LlamaBlock(MASTER_CONFIG))
    block_states = [benchmark_block.get_state() for _ in stubs]
    setup_promises = [stub.net.push_block(state) for stub, state in zip(stubs, block_states)]
    setup_scores = await asyncio.gather(*setup_promises)

    print('benchmarking!')
    benchmark_promises = [benchmark(stub, torch.randn([32, 16, MASTER_CONFIG['d_model']])) for stub in stubs]
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
    for batch_index in range(1):

        batch = torch.randn([num_mini_batches, 32, 16, MASTER_CONFIG['d_model']])
        target = torch.randn([32, 16, MASTER_CONFIG['d_model']])

        def get_pipeline_stages(j, x):
            pipeline_stages = []
            ctx_id = uuid.uuid4()

            for _i in range(len(stubs)):

                async def next_stage(i=_i):

                    if (i == 0): await stubs[i].load_activations(ctx_id, x.clone())
                    if (i != 0): await stubs[i].fetch_activations(ctx_id, urls[i-1])
                    
                    if (i != len(stubs)-1):
                        await stubs[i].forward(ctx_id)
                    else:
                        await stubs[i].final_stage(ctx_id, target, criterion_str)

                pipeline_stages.append(metrics_wrapper(f"f{_i+1}s{j+1}", next_stage()))

            for _i in range(len(stubs)-1, -1, -1):

                async def next_stage(i=_i):

                    if (i != len(stubs)-1): await stubs[i].fetch_gradients(ctx_id, urls[i+1], clear_cache=True)
                    await stubs[i].backward(ctx_id, clear_cache=True)

                pipeline_stages.append(metrics_wrapper(f"b{_i+1}s{j+1}", next_stage()))

            return pipeline_stages

        flow = get_pipeline_parallel_flow(num_workers, get_pipeline_stages, batch)

        print('executing training flow')
        await flow.start()

        print('optimizer step')
        await multi_block_stub.step([{"zero_grad": True} for _ in stubs])

        print("plotting!")
        plot_timings()

if (__name__ == "__main__"): asyncio.run(main())