import axon
import asyncio
import torch
import threading
import time
import pickle

from VGG_blocks import VGGBlock_1

tensors = []

@axon.worker.rpc()
def save_tensors(x):
	global tensors
	tensors.append(x)

async def main():

	worker_thread = threading.Thread(target=axon.worker.init, daemon=True)
	worker_thread.start()
	time.sleep(0.1)

	net = VGGBlock_1()

	x = torch.randn([10, 3, 32, 32])

	x, skip = net(x)

	x = x.clone()
	skip = skip.clone()

	print(x.shape)
	x_str = pickle.dumps(x)
	print(len(x_str))

	stub = axon.client.get_stub('localhost', stub_type=axon.stubs.SyncStub)
	stub.rpc.save_tensors(x)

	# print(tensors)

asyncio.run(main())