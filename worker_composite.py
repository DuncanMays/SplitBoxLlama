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

# this class accepts a list of axon.client.RemoteWorker objects and returns an object with all the RPCs that are common between those worker handles
# RPCs on all workers can then be called with a single function call on an instance of this class, making cluster management much smoother and easier
class WorkerComposite():

	def __init__(self, children):
		self.children = children
		self.rpcs = SimpleNamespace(**{})

		self.compose_RPCs()

	def compose_RPCs(self):
		# the names of each RPC on each child
		# child_RPC_names = [list(child.rpcs.__dict__) for child in self.children]
		child_RPC_names = [list(dir(child.rpcs)) for child in self.children]
		# The RPCs that this class offers are the intersection between the RPCs the children offer
		self_RPC_names = reduce_intersection(child_RPC_names)
		# the object that holds the stubs for the RPCs of each child, wrapped in asyncio gather calls
		rpcs = {}

		for RPC_name in self_RPC_names:
			# the coroutines that call RPCs called RPC_name on each worker independantly
			child_RPC_stubs = [getattr(child.rpcs, RPC_name) for child in self.children]
			# gather_wrapper wraps the RPC stubs with an asyncio.gather call, which allows them to execute concurrently
			rpcs[RPC_name] = gather_wrapper(child_RPC_stubs)

		self.rpcs = SimpleNamespace(**rpcs)