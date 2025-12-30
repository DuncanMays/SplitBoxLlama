import pytest
import torch

from pipeline_client import benchmark, get_training_flow

from SplitBox.worker import Worker, NeuralBlock
from SplitBox.worker_test import make_net, mock_worker_factory
from SplitBox.multi_stub import async_wrapper, sync_wrapper

# takes a set of worker objects and URLs
# returns a list of mock stub objects where requests in between workers pass through local, syn objects
# returns a mock list of stubs
def get_mock_cluster(workers, urls):
	async_stubs = [None]*len(workers)
	sync_stubs = [None]*len(workers)

	# create async and sync stubs for each worker object
	for i in range(len(workers)): 
		worker = workers[i]
		async_stubs[i] = async_wrapper(worker)
		sync_stubs[i] = sync_wrapper(worker)

	# set the stub cache for each worker to point to sync stubs of other workers
	stub_cache = {url: sync_stub for url, sync_stub in zip(urls, sync_stubs)}
	for inner_stub in workers:
		inner_stub.stub_cache = stub_cache

	return async_stubs

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
	skip = torch.randn([32, 20], requires_grad=True)
	skip.retain_grad()
	target = torch.randn([32, 20], requires_grad=True)
	target.retain_grad()

	criterion = torch.nn.functional.cross_entropy

	urls = [f'url_{i}' for i in range(num_workers)]
	workers = [mock_worker_factory() for i in range(num_workers)]
	async_stubs = get_mock_cluster(workers, urls)

	flow, losses = get_training_flow(async_stubs, urls, batch, target, criterion)
	await flow.start()

# @pytest.mark.asyncio
# async def test_multiple_activations():
# 	num_workers = 3

# 	batch = torch.randn([32, 20], requires_grad=True)
# 	batch.retain_grad()
# 	skip = torch.randn([32, 20], requires_grad=True)
# 	skip.retain_grad()
# 	target = torch.randn([32, 20], requires_grad=True)
# 	target.retain_grad()

# 	criterion = torch.nn.functional.cross_entropy

# 	urls = [f'url_{i}' for i in range(num_workers)]

# 	workers = [
# 		mock_worker_factory(net_factory=OneTwoNet),
# 		mock_worker_factory(net_factory=TwoTwoNet),
# 		mock_worker_factory(net_factory=TwoOneNet)
# 		]

# 	async_stubs = get_mock_cluster(workers, urls)

# 	flow, losses = get_training_flow(async_stubs, urls, batch, target, criterion)
# 	await flow.start()