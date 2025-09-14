import pytest
import torch

from worker import NeuralBlock

def make_net():

	ffn = torch.nn.Sequential(
		torch.nn.Linear(20, 100),
		torch.nn.Linear(100, 20),
	)

	return ffn

def test_NeuralBlock():

	block = NeuralBlock(make_net)

	x = torch.randn([32, 20])
	y = block(x)

def test_backprop():

	block = NeuralBlock(make_net)

	x = torch.randn([32, 20])

	y = block(x)
	start_loss = torch.sum(torch.abs(y))

	for i in range(100):

		y = block(x)

		loss = torch.sum(torch.abs(y))

		block.optimizer.zero_grad()
		loss.backward()
		block.optimizer.step()

	y = block(x)
	end_loss = torch.sum(torch.abs(y))

	assert(start_loss > end_loss)

def test_serialization():

	block = NeuralBlock(make_net)

	x = torch.randn([32, 20])

	start_y = block(x)

	state = block.get_state()
	
	new_block = NeuralBlock(make_net)
	new_block.set_state(state)

	end_y = new_block(x)

	assert(torch.equal(start_y, end_y))

def test_serialization_backprop():

	block = NeuralBlock(make_net)
	state = block.get_state()
	
	new_block = NeuralBlock.from_state(state)

	new_block = NeuralBlock(make_net)

	x = torch.randn([32, 20])

	y = new_block(x)
	start_loss = torch.sum(torch.abs(y))

	for i in range(100):

		y = new_block(x)

		loss = torch.sum(torch.abs(y))

		new_block.optimizer.zero_grad()
		loss.backward()
		new_block.optimizer.step()

	y = new_block(x)
	end_loss = torch.sum(torch.abs(y))

	assert(start_loss > end_loss)