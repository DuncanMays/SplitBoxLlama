import asyncio
import torch
import uuid

from pipeline_parallel import get_pipeline_parallel_flow
from plot_pipeline import fake_task
from worker import device

# from worker.py
class MockNeuralBlock():

    def __init__(self, gradman, device=device):
        pass

    async def run_net(self, x, URL, direction, call_id, return_outputs=False, clear_local_cache=False, clear_remote_cache=False, save_tensors=False):
        pass
            
    def get_outputs(self, call_id, clear_cache=False):
        pass

async def main():

    batch = torch.randn([32, 28, 28])

    stub_1 = MockNeuralBlock(None)
    stub_2 = MockNeuralBlock(None)
    stub_3 = MockNeuralBlock(None)

    def get_pipeline_stages(j, x):
        pipeline_stages = []
        ctx_id = uuid.uuid4()

        pipeline_stages.append(stub_1.run_net(x, None, 'forward', ctx_id, save_tensors=True))
        pipeline_stages.append(stub_2.run_net(None, "url_1", 'forward', ctx_id, save_tensors=True))
        pipeline_stages.append(stub_3.run_net(None, "url_2", 'forward', ctx_id, save_tensors=True, return_outputs=True))

        pipeline_stages.append(stub_3.run_net(x, None, 'backward', ctx_id, clear_remote_cache=True))
        pipeline_stages.append(stub_2.run_net(None, "url_3", 'backward', ctx_id, clear_remote_cache=True))
        pipeline_stages.append(stub_1.run_net(None, "url_2", 'backward', ctx_id, clear_remote_cache=True, return_outputs=True, clear_local_cache=True))

        return pipeline_stages

    flow = get_pipeline_parallel_flow(3, get_pipeline_stages, batch)

    await flow.start()

asyncio.run(main())