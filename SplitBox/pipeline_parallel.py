import asyncio
from .events import EventEmitter


class EventFlow:
	def __init__(self):
		self._emitter = EventEmitter()
		self._fired = set()   # names of events that have already been emitted
		self._pending = []    # (triggers, callbacks, events) queued before start()
		self._started = False
		self._tasks = set()

	def _launch(self, coro):
		task = asyncio.create_task(coro)
		self._tasks.add(task)

	def _schedule_action(self, triggers, callbacks, events):
		fired_so_far = set()

		async def run():
			await asyncio.gather(*callbacks)
			for name in events:
				self._fired.add(name)
				self._emitter.emit(name)

		def check_and_launch():
			if fired_so_far >= set(triggers):
				self._launch(run())

		def make_listener(trigger):
			def listener():
				fired_so_far.add(trigger)
				check_and_launch()
			return listener

		if not triggers:
			self._launch(run())
			return

		for trigger in triggers:
			if trigger in self._fired:
				fired_so_far.add(trigger)
			else:
				self._emitter.on(trigger, make_listener(trigger))

		check_and_launch()

	def set_action(self, triggers, callbacks, events):
		if self._started:
			self._schedule_action(triggers, callbacks, events)
		else:
			self._pending.append((triggers, callbacks, events))

	async def _join(self):
		while self._tasks:
			snapshot = list(self._tasks)
			await asyncio.gather(*snapshot, return_exceptions=True)
			for t in snapshot:
				self._tasks.discard(t)

	async def start(self):
		self._started = True
		for triggers, callbacks, events in self._pending:
			self._schedule_action(triggers, callbacks, events)
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