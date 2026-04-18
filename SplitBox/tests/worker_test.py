import pytest
import torch
import pickle

from SplitBox.worker import NeuralBlock, BlockStack, Worker
from SplitBox.multi_stub import async_wrapper, sync_wrapper
from SplitBox.SplitNets import OneTwoNet, TwoOneNet

def make_net():

	ffn = torch.nn.Sequential(
		torch.nn.Linear(20, 100),
		torch.nn.Linear(100, 20),
	)

	return ffn

def mock_worker_factory(net_factory=make_net):
	nb = NeuralBlock(net_factory)
	stack = BlockStack()
	stack.push_block(nb.get_state())
	mock_worker = Worker(stack)

	return mock_worker

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

def test_BlockStack():

	blocks = [NeuralBlock(make_net) for _ in range(5)]
	block_states = [block.get_state() for block in blocks]

	stack = BlockStack()
	stack.load_blocks(block_states)

	x = torch.randn([32, 20])

	y = stack(x)
	start_loss = torch.sum(torch.abs(y))

	for i in range(100):

		y = stack(x)

		loss = torch.sum(torch.abs(y))

		stack.zero_grad()
		loss.backward()
		stack.step()

	y = stack(x)
	end_loss = torch.sum(torch.abs(y))

	assert(start_loss > end_loss)

def test_training_loop():
	num_workers = 5

	A = mock_worker_factory()
	B = mock_worker_factory()

	A.stub_cache = {"B": sync_wrapper(B)}
	B.stub_cache = {"A": sync_wrapper(A)}

	ctx_id = 'worker_test.test_training_loop'
	x = torch.randn([32, 20])
	y = torch.randn([32, 20])
	criterion_str = pickle.dumps(torch.nn.functional.mse_loss)

	losses = []

	for _ in range(10):
		A.load_activations(ctx_id, x)
		A.forward(ctx_id)

		B.fetch_activations(ctx_id, "A")
		
		loss = B.final_stage(ctx_id, y, criterion_str)
		losses.append(loss)

		B.backward(ctx_id)

		A.fetch_gradients(ctx_id, "B")
		A.backward(ctx_id)

		B.net.step(zero_grad=True)
		A.net.step(zero_grad=True)

	assert(losses[0] > losses[-1])

def test_multiple_activations():
	num_workers = 5

	A = mock_worker_factory(net_factory=OneTwoNet)
	B = mock_worker_factory(net_factory=TwoOneNet)

	A.stub_cache = {"B": sync_wrapper(B)}
	B.stub_cache = {"A": sync_wrapper(A)}

	ctx_id = 'worker_test.test_training_loop'
	x = torch.randn([32, 20])
	y = torch.randn([32, 20])
	criterion_str = pickle.dumps(torch.nn.functional.mse_loss)

	losses = []

	for _ in range(10):
		A.load_activations(ctx_id, x)
		A.forward(ctx_id)

		B.fetch_activations(ctx_id, "A")
		
		loss = B.final_stage(ctx_id, y, criterion_str)
		losses.append(loss)

		B.backward(ctx_id)

		A.fetch_gradients(ctx_id, "B")
		A.backward(ctx_id)

		B.net.step(zero_grad=True)
		A.net.step(zero_grad=True)

	assert(losses[0] > losses[-1])

def test_custom_optimizer():
	optimizer_fn = lambda params: torch.optim.SGD(params, lr=0.001, momentum=0.9)
	block = NeuralBlock(make_net, optimizer_fn=optimizer_fn)

	assert isinstance(block.optimizer, torch.optim.SGD)

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

def test_scheduler_serialization():
	optimizer_fn = lambda params: torch.optim.SGD(params, lr=0.1)
	scheduler_fn = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
	block = NeuralBlock(make_net, optimizer_fn=optimizer_fn, scheduler_fn=scheduler_fn)

	for _ in range(3):
		block.scheduler.step()

	lr_before = block.optimizer.param_groups[0]['lr']
	last_epoch_before = block.scheduler.last_epoch

	state = block.get_state()
	restored = NeuralBlock.from_state(state)

	assert restored.scheduler.last_epoch == last_epoch_before
	assert abs(restored.optimizer.param_groups[0]['lr'] - lr_before) < 1e-6

def test_scheduler_step_on_stack():
	optimizer_fn = lambda params: torch.optim.SGD(params, lr=0.1)
	scheduler_fn = lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
	block = NeuralBlock(make_net, optimizer_fn=optimizer_fn, scheduler_fn=scheduler_fn)

	stack = BlockStack()
	stack.push_block(block.get_state())

	lr_before = stack.blocks[0].optimizer.param_groups[0]['lr']
	stack.scheduler_step()
	lr_after = stack.blocks[0].optimizer.param_groups[0]['lr']

	assert lr_after < lr_before

def make_dropout_net():
	return torch.nn.Sequential(
		torch.nn.Linear(20, 100),
		torch.nn.Dropout(p=0.5),
		torch.nn.Linear(100, 20),
	)

def test_train_mode_on_stack():
	block = NeuralBlock(make_dropout_net)
	stack = BlockStack()
	stack.push_block(block.get_state())

	stack.train_mode(True)
	assert stack.blocks[0].net.training == True

	stack.train_mode(False)
	assert stack.blocks[0].net.training == False

def test_train_mode_on_worker():
	block = NeuralBlock(make_dropout_net)
	stack = BlockStack()
	stack.push_block(block.get_state())
	worker = Worker(stack)

	worker.train_mode(True)
	assert stack.blocks[0].net.training == True

	worker.train_mode(False)
	assert stack.blocks[0].net.training == False

def test_inference_forward_no_grad():
	A = mock_worker_factory()

	ctx_id = 'test_inference_forward_no_grad'
	x = torch.randn([32, 20])

	A.load_activations(ctx_id, x)
	A.forward(ctx_id, inference=True)
	y_infer = A.get_activations(ctx_id)[0]

	assert y_infer.grad_fn is None

	A.load_activations(ctx_id, x)
	A.forward(ctx_id, inference=False)
	y_train = A.get_activations(ctx_id)[0]

	assert y_train.grad_fn is not None

def test_eval_flow_output_shape():
	A = mock_worker_factory()
	B = mock_worker_factory()

	A.stub_cache = {"B": sync_wrapper(B)}
	B.stub_cache = {"A": sync_wrapper(A)}

	ctx_id = 'test_eval_flow_output_shape'
	x = torch.randn([32, 20])

	A.load_activations(ctx_id, x)
	A.forward(ctx_id, inference=True)

	B.fetch_activations(ctx_id, "A")
	B.forward(ctx_id, inference=True)

	y = B.get_activations(ctx_id)
	assert y[0].shape == torch.Size([32, 20])

def test_eval_flow_no_grad():
	A = mock_worker_factory()
	B = mock_worker_factory()

	A.stub_cache = {"B": sync_wrapper(B)}
	B.stub_cache = {"A": sync_wrapper(A)}

	ctx_id = 'test_eval_flow_no_grad'
	x = torch.randn([32, 20])

	A.load_activations(ctx_id, x)
	A.forward(ctx_id, inference=True)

	B.fetch_activations(ctx_id, "A")
	B.forward(ctx_id, inference=True)

	y = B.get_activations(ctx_id)
	assert y[0].grad_fn is None

def test_eval_flow_matches_local():
	x = torch.randn([32, 20])

	block_a = NeuralBlock(make_net)
	block_b = NeuralBlock(make_net)

	stack_a = BlockStack()
	stack_a.push_block(block_a.get_state())
	A = Worker(stack_a)

	stack_b = BlockStack()
	stack_b.push_block(block_b.get_state())
	B = Worker(stack_b)

	A.stub_cache = {"B": sync_wrapper(B)}
	B.stub_cache = {"A": sync_wrapper(A)}

	ctx_id = 'test_eval_flow_matches_local'
	A.load_activations(ctx_id, x)
	A.forward(ctx_id, inference=True)
	B.fetch_activations(ctx_id, "A")
	B.forward(ctx_id, inference=True)
	pipeline_out = B.get_activations(ctx_id)[0]

	with torch.no_grad():
		mid = stack_a.blocks[0].net(x)
		local_out = stack_b.blocks[0].net(mid)

	assert torch.allclose(pipeline_out, local_out)

def test_eval_flow_multiple_activations():
	A = mock_worker_factory(net_factory=OneTwoNet)
	B = mock_worker_factory(net_factory=TwoOneNet)

	A.stub_cache = {"B": sync_wrapper(B)}
	B.stub_cache = {"A": sync_wrapper(A)}

	ctx_id = 'test_eval_flow_multiple_activations'
	x = torch.randn([32, 20])

	A.load_activations(ctx_id, x)
	A.forward(ctx_id, inference=True)

	B.fetch_activations(ctx_id, "A")
	B.forward(ctx_id, inference=True)

	y = B.get_activations(ctx_id)
	assert y[0].shape == torch.Size([32, 20])

def test_eval_flow_no_backward_side_effects():
	A = mock_worker_factory()
	B = mock_worker_factory()

	A.stub_cache = {"B": sync_wrapper(B)}
	B.stub_cache = {"A": sync_wrapper(A)}

	ctx_id = 'test_eval_flow_no_backward_side_effects'
	x = torch.randn([32, 20])

	A.load_activations(ctx_id, x)
	A.forward(ctx_id, inference=True)

	B.fetch_activations(ctx_id, "A")
	B.forward(ctx_id, inference=True)

	assert len(A.saved_output_grads) == 0
	assert len(A.saved_input_grads) == 0
	assert len(B.saved_output_grads) == 0
	assert len(B.saved_input_grads) == 0
