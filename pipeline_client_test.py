import pytest
import torch

from worker import Worker, NeuralBlock
from worker_test import make_net
from multi_stub import async_wrapper, sync_wrapper
from pipeline_client import benchmark, get_training_flow

def mock_worker_factory():
	nb = NeuralBlock(make_net)
	mock_worker = Worker(nb)

	return mock_worker

@pytest.mark.asyncio
async def test_benchmark():
	x = torch.randn([32, 20])
	worker = mock_worker_factory()
	worker = async_wrapper(worker)
	score = await benchmark(worker, x)

	print(score)

@pytest.mark.asyncio
async def test_training_flow():
	num_workers = 3

	batch = torch.randn([32, 20], requires_grad=True)
	batch.retain_grad()

	target = torch.randn([20], requires_grad=True)
	target.retain_grad()

	urls = [f'url_{i}' for i in range(num_workers)]
	inner_stubs = [mock_worker_factory() for i in range(num_workers)]
	async_stubs = [_ for _ in range(num_workers)]
	sync_stubs = [_ for _ in range(num_workers)]

	# stub_cache = {url: mock_stub for url, mock_stub in zip(urls, mock_stubs)}
	
	for i in range(num_workers): 
		inner_stub = inner_stubs[i]
		async_stubs[i] = async_wrapper(inner_stub)
		sync_stubs[i] = sync_wrapper(inner_stub)

	stub_cache = {url: sync_stub for url, sync_stub in zip(urls, sync_stubs)}
	for inner_stub in inner_stubs:
		inner_stub.stub_cache = stub_cache

	flow = get_training_flow(urls, async_stubs, batch, target)

	await flow.start()