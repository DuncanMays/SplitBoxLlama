import pytest
import axon
import asyncio
import time

from threading import Thread

from SplitBox.multi_stub import get_multi_stub

def async_wrapper(obj):

	elements = dir(obj)
	attrs = {}

	def _async_wrapper(fn):

		async def async_fn(*args, **kwargs):
			await asyncio.sleep(0)
			param_str = axon.serializers.serialize((args, kwargs))
			new_args, new_kwargs = axon.serializers.deserialize(param_str)
			result = fn(*new_args, **new_kwargs)
			result_str = axon.serializers.serialize(result)
			return axon.serializers.deserialize(result_str)

		return async_fn

	for child_name in elements:
		attrs[child_name] = _async_wrapper(getattr(obj, child_name))

	return type('ServiceStub', (), attrs)

def sync_wrapper(obj):

	elements = dir(obj)
	attrs = {}

	def _sync_wrapper(fn):

		def sync_fn(*args, **kwargs):
			param_str = axon.serializers.serialize((args, kwargs))
			new_args, new_kwargs = axon.serializers.deserialize(param_str)
			result = fn(*new_args, **new_kwargs)
			result_str = axon.serializers.serialize(result)
			return axon.serializers.deserialize(result_str)

		return sync_fn

	for child_name in elements:
		attrs[child_name] = _sync_wrapper(getattr(obj, child_name))

	return type('ServiceStub', (), attrs)

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