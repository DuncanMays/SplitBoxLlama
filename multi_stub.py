import axon
import asyncio
from types import SimpleNamespace

# returns a list with the intersection of elements from lists l1 and l2
def list_intersection(l1, l2):
	intersection = []
	for l in l1:
		if l in l2:
			intersection.append(l)

	return intersection

# takes a list of lists L, and returns the list that is the intersection between all lists in L
def reduce_intersection(L):
	L = iter(L)
	intersection = next(L)

	for l in L:
		intersection = list_intersection(intersection, l)

	return intersection

# takes a list of coroutines and calls them in a gather statement
def gather_wrapper(child_coros):
	num_children = len(child_coros)
	
	async def gathered_coroutine(param_list=None):
		# if no param_list is specified, then child_coros probably don't take any parameters, therefor fill param_list with empty parameter tuples
		if (param_list == None):
			param_list = [() for i in child_coros]

		elif (num_children != len(param_list)):
			raise BaseException('given coroutine list and parameter list not the same length')

		tasks = []
		for i in range(num_children):
			coro = child_coros[i]
			params = param_list[i]

			tasks.append(coro(*params))

		return await asyncio.gather(*tasks)

	return gathered_coroutine

# this function accepts a list of axon.client.ServiceStub objects (called elements) and returns a stub with all child stubs that are common between those stubs
# RPCs on all workers can then be called with a single function call on an instance of this class, making cluster management much smoother and easier
def get_multi_stub(elements):
	# the names of the children on each element stub
	element_child_names = [list(dir(child)) for child in elements]
	# The child stubs that this stub offers are the common stubs between all the elements
	child_names = reduce_intersection(element_child_names)
	# print("child_names")
	# print(child_names)
	# the object that holds the stubs for the elemets, wrapped in asyncio gather calls
	attrs = {}

	for child_name in child_names:
		# the coroutines that call RPCs called child_name on each worker independantly
		child_RPC_stubs = [getattr(e, child_name) for e in elements]
		# gather_wrapper wraps the RPC stubs with an asyncio.gather call, which allows them to execute concurrently
		attrs[child_name] = gather_wrapper(child_RPC_stubs)

	return type('ServiceStub', (), attrs)

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