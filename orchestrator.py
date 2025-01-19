import axon
import asyncio

async def main():

	url_1 = "192.168.2.19:8001/block"
	url_2 = "192.168.2.44:8001/block"

	stub_1 = axon.client.get_stub(url_1)
	stub_2 = axon.client.get_stub(url_2)

	await stub_1.forward("hello!", None, 100)
	msg = await stub_2.forward(None, url_1, 100, return_msg=True)
	
	print(msg)

asyncio.run(main())