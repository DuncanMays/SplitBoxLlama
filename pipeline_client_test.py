import pytest
import torch

from worker import Worker, NeuralBlock
from worker_test import make_net
from multi_stub import async_wrapper
from pipeline_client import benchmark

def mock_worker_factory():
	nb = NeuralBlock(make_net)
	return Worker(nb)

@pytest.mark.asyncio
async def test_benchmark():
	x = torch.randn([32, 20])
	worker = mock_worker_factory()
	worker  = async_wrapper(worker)
	score = await benchmark(worker, x)

	print(score)

# def test_main():

# 	workers = [mock_worker_factory() for _ in range(5)]