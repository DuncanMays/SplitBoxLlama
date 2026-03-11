import pytest
import axon
import time

from threading import Thread

from SplitBox.multi_stub import get_multi_stub

@pytest.fixture(scope="package")
def triple_worker():

	class ReturnValue:

		def __init__(self, value):
			self.value = value

		def get_value(self):
			return self.value

	axon.worker.service(ReturnValue(1), 'one')
	axon.worker.service(ReturnValue(2), 'two')
	axon.worker.service(ReturnValue(3), 'three')

	worker_thread = Thread(target=axon.worker.init, daemon=True)
	worker_thread.start()

	time.sleep(0.5)

@pytest.fixture()
def triple_stubs(triple_worker):

	stub_1 = axon.client.get_stub('localhost/one')
	stub_2 = axon.client.get_stub('localhost/two')
	stub_3 = axon.client.get_stub('localhost/three')

	return stub_1, stub_2, stub_3

@pytest.mark.asyncio
async def test_basic(triple_stubs):
	worker_composite = get_multi_stub(triple_stubs)
	assert(await worker_composite.get_value() == [1, 2, 3])