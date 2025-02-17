import axon
import asyncio
import torch
import uuid

async def main():

	call_id = "long_random_string"

	url_1 = "localhost:8001/block"
	url_2 = "localhost:8002/block"

	print('creating stubs')
	stub_1 = axon.client.get_stub(url_1, stub_type=axon.stubs.SyncStub)
	stub_2 = axon.client.get_stub(url_2, stub_type=axon.stubs.SyncStub)

	print('FnStub')
	class FnStub(torch.autograd.Function):

		def __init__(self):
			super().__init__()

		@staticmethod
		def forward(ctx, x):

			# if the context already has an ID, that means it's been through a FnStub already
			# this could be a problem for recursive patterns, in that case, the stub's context store will need to be a dict of lists of contexts
			if not hasattr(ctx, 'id'):
				ctx.id = uuid.uuid4()

			stub_1.apply(x, None, ctx.id)
			x = stub_2.apply(None, url_1, ctx.id, return_outputs=True)

			return x

		@staticmethod
		def backward(ctx, g):
			stub_2.apply_gradients(g, None, ctx.id)
			g = stub_1.apply_gradients(None, url_2, ctx.id, return_outputs=True)
			return g

	print('instantiating FnStub')
	fn = FnStub()

	in_x = torch.randn([32, 100], requires_grad=True)

	x = fn.apply(in_x)
	loss = x.sum()
	print(in_x.grad)
	loss.backward()
	print(in_x.grad)

print('calling main')
asyncio.run(main())