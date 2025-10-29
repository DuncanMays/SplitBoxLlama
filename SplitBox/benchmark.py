import time
import asyncio
import torch

class MockWorker():

    def __init__(self, num_blocks=10, forward_rate=10, backward_rate=8, download_rate=100, activation_size=1):
        self.num_blocks = num_blocks
        self.forward_rate = forward_rate
        self.backward_rate = backward_rate
        self.download_rate = download_rate
        self.activation_size = activation_size

    def set_blocks(self, num_blocks):
        self.num_blocks = num_blocks

    async def forward(self, activation_id, clear_cache=False):
        await asyncio.sleep(0.01*self.num_blocks/self.forward_rate)

    async def backward(self, activation_id, clear_cache=False):
        await asyncio.sleep(0.01*self.num_blocks/self.backward_rate)

    async def get_activations(self, activation_id, clear_cache=False):
        await asyncio.sleep(self.activation_size/self.download_rate)

    async def get_gradients(self, activation_id, clear_cache=False):
        await asyncio.sleep(self.activation_size/self.download_rate)

    async def load_activations(self, activation_id, x):
        await asyncio.sleep(x.numel()/(1000*self.download_rate))

    async def load_gradients(self, activation_id, g):
        await asyncio.sleep(g.numel()/(1000*self.download_rate))

    async def fetch_activations(self, activation_id, source_URL, clear_cache=False):
        await asyncio.sleep(self.activation_size/self.download_rate)

    async def fetch_gradients(self, activation_id, source_URL, clear_cache=False):
        await asyncio.sleep(self.activation_size/self.download_rate)

# returns the training rate of the worker in seconds/block, as well as the time to load a batch
# ie, multiply training rate by the number of blocks to get the number of seconds an operation should take
async def benchmark(worker, x, num_blocks):
    worker.set_blocks(num_blocks)
    start = time.time()

    await worker.load_activations('activation_id', x)

    net_time = time.time() - start
    start = time.time()

    await worker.forward('activation_id')

    comp_rate = (time.time() - start)/num_blocks
    return comp_rate, net_time

async def main():
    
    worker = MockWorker()

    benchmarks = await benchmark(worker, torch.randn([32, 28, 28]), 20)

    print(benchmarks)

if (__name__ == "__main__"): asyncio.run(main())