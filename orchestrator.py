import axon
import asyncio
import torch

async def main():

	call_id = "long_random_string"

	url_1 = "localhost:8001/block"
	url_2 = "localhost:8002/block"

	stub_1 = axon.client.get_stub(url_1)
	stub_2 = axon.client.get_stub(url_2)

	ln_1 = torch.nn.Linear(10, 100)
	ln_2 = torch.nn.Linear(100, 10)

	x = torch.randn([32, 10])

	x = ln_1(x)

	await stub_1.apply(x, None, call_id)
	x = await stub_2.apply(None, url_1, call_id, return_outputs=True)

	x = ln_2(x)

	loss = x.sum()
	loss.backward()
	
	print(loss)

asyncio.run(main())