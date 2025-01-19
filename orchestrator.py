import axon
import asyncio

async def main():
	stub_1 = axon.client.get_stub("localhost:8001")
	# stub_2 = axon.client.get_stub(":8001")

	await stub_1.block.forward("hello!", None, 100)
	msg = await stub_1.block.get_activations(100)
	print(msg)

asyncio.run(main())