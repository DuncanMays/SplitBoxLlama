import asyncio

class EventFlow():
	"""
	A dynamic async dependency graph executor.

	Actions (trigger names -> callbacks -> output event names) can be
	registered at any time: before start(), after start(), or from within
	a running callback. This makes it suitable for long-running server-side
	use where the graph grows continuously.

	Memory management
	-----------------
	- asyncio.Event objects are released once they have fired and every
	  registered waiter has been unblocked (tracked via _waiter_counts).
	- asyncio.Task handles are released immediately on completion via a
	  done-callback.
	- Fired event *names* (plain strings) are kept in _fired indefinitely
	  so that late-arriving set_action calls referencing an already-fired
	  event proceed immediately rather than hanging. The memory cost is
	  O(unique event names), which is negligible for string data.

	Not thread-safe. All calls must originate from within the same asyncio
	event loop.
	"""

	def __init__(self):
		self._events = {}        # name -> asyncio.Event (released when no longer needed)
		self._waiter_counts = {} # name -> int, number of pending waiters
		self._fired = set()      # names of events that have fired
		self._tasks = set()      # live asyncio.Task handles
		self._pending = []       # coroutines queued before start()
		self._started = False

	def _get_or_create_event(self, name):
		if name not in self._events:
			e = asyncio.Event()
			if name in self._fired:
				e.set()  # late arrival — already fired, unblock immediately
			self._events[name] = e
			self._waiter_counts[name] = 0
		return self._events[name]

	def _try_cleanup_event(self, name):
		"""Release the Event object once it has fired and has no remaining waiters."""
		if (name in self._waiter_counts
				and self._waiter_counts[name] == 0
				and name in self._fired):
			del self._events[name]
			del self._waiter_counts[name]

	async def _run_action(self, trigger_names, callbacks, event_names):
		# Capture the task handle now so we can remove it from _tasks before
		# the task is marked done. This ensures _join sees an accurate count
		# the moment asyncio.gather resolves — avoiding an infinite while loop
		# that would occur if done-callbacks ran too late.
		current_task = asyncio.current_task()
		try:
			for name in trigger_names:
				await self._events[name].wait()
				self._waiter_counts[name] -= 1
				self._try_cleanup_event(name)

			await asyncio.gather(*callbacks)

			for name in event_names:
				self._get_or_create_event(name)
				self._fired.add(name)
				self._events[name].set()
				self._try_cleanup_event(name)
		finally:
			self._tasks.discard(current_task)

	def get_event(self, name):
		return self._get_or_create_event(name)

	def set_action(self, triggers, callbacks, events):
		for name in triggers:
			self._get_or_create_event(name)
			self._waiter_counts[name] += 1

		coro = self._run_action(triggers, callbacks, events)

		if self._started:
			task = asyncio.ensure_future(coro)
			self._tasks.add(task)
		else:
			self._pending.append(coro)

	async def _join(self):
		"""Wait until there are no live tasks. Re-checks after each gather
		to catch tasks that were added by callbacks during execution."""
		while self._tasks:
			await asyncio.gather(*list(self._tasks), return_exceptions=True)

	async def start(self):
		self._started = True
		for coro in self._pending:
			task = asyncio.ensure_future(coro)
			self._tasks.add(task)
			task.add_done_callback(self._tasks.discard)
		self._pending.clear()
		await self._join()

def parse_task_str(task_str):
	d = task_str[0]

	l = task_str.split('s')
	w = int(l[0][1:])
	s = int(l[1])

	return d, w, s

# returns the prior task on a worker in a pipeline parallel training schedule
def get_prior_task(task_str, num_workers, num_stages):
	
	d, w, s = parse_task_str(task_str)

	prior_d = None
	prior_s = None

	if (d == 'f'):
		prior_s = s - ( num_workers - w + 1 )

		if (prior_s < 1):
			prior_s = s - 1
			prior_d = 'f'

		else:
			prior_d = 'b'

	elif (d == 'b'):
		prior_s = s + ( num_workers - w )

		if (prior_s > num_stages):
			prior_s = s - 1
			prior_d = 'b'

		else:
			prior_d = 'f'

	else:
		raise BaseException(f'unrecognized direction code: {d}')

	return f'{prior_d}{w}s{prior_s}'

def get_pipeline_parallel_flow(num_workers, get_pipeline_stages, batch, target):
	
	pipeline_prefixes = []
	for i in range(num_workers): pipeline_prefixes.append(f'f{i+1}')
	for i in range(num_workers): pipeline_prefixes.append(f'b{num_workers-i}')

	# tasks are indicated with a letter and two numbers
	# the letter, either f or b means forward or backwards
	# the first number indicates the worker
	# the second number indicates the batch

	flow = EventFlow()

	for minibatch_index in range(len(batch)):

		minibatch = batch[minibatch_index]
		minibatch_target = target[minibatch_index]

		pipeline_stages = get_pipeline_stages(minibatch_index, minibatch, minibatch_target)

		if ((len(pipeline_stages) // num_workers) != 2):
			raise BaseException(f"Incompatible number of pipeline stages and workers: {len(pipeline_stages)}, {num_workers}")

		for i, prefix in enumerate(pipeline_prefixes):

			task_str = f'{prefix}s{minibatch_index+1}'

			triggers = []

			# if not the first stage, add the prior task on the worker as a trigger
			if (minibatch_index != 0):
				triggers.append(get_prior_task(task_str, num_workers, len(batch)))

			# if not the first pipeline step, add the prior pipeline step as a trigger
			if (i != 0):
				prior_prefix = pipeline_prefixes[i-1]
				triggers.append(f'{prior_prefix}s{minibatch_index+1}')

			callbacks = [pipeline_stages[i]]

			events = [task_str]

			flow.set_action(triggers, callbacks, events)

	return flow