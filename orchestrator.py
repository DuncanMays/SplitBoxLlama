import axon
import asyncio

async def main():
	stub = axon.client.get_stub("localhost:8001")
	await stub.block.forward("hello!", 100)
	msg = await stub.block.get_activations(100)
	print(msg)

asyncio.run(main())