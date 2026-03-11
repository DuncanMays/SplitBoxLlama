import asyncio
from .EventEmitter import EventEmitter

class EventFlow:
	def __init__(self):
		self._emitter = EventEmitter()
		self._fired = set()   # names of events that have already been emitted
		self._pending = []    # (triggers, callbacks, events) queued before start()
		self._started = False
		self._tasks = set()
		self._trigger_refs = {}  # trigger -> count of actions not yet scheduled that reference it

	def _add_to_tasks(self, coro):
		task = asyncio.create_task(coro)
		self._tasks.add(task)

	def _schedule_action(self, triggers, callbacks, events):
		fired_so_far = set()

		async def run_callbacks():
			await asyncio.gather(*callbacks)

			for name in events:
				self._fired.add(name)
				self._emitter.emit(name)

		def check_and_launch():

			if fired_so_far >= set(triggers):
				self._add_to_tasks(run_callbacks())

				for trigger in triggers:
					self._trigger_refs[trigger] -= 1

					if self._trigger_refs[trigger] <= 0:
						del self._trigger_refs[trigger]
						self._fired.discard(trigger)

		def make_listener(trigger):

			def listener():
				fired_so_far.add(trigger)
				check_and_launch()

			return listener

		if not triggers:
			self._add_to_tasks(run_callbacks())
			return

		for trigger in triggers:

			if trigger in self._fired:
				fired_so_far.add(trigger)

			else:
				self._emitter.on(trigger, make_listener(trigger))

		check_and_launch()

	def set_action(self, triggers, callbacks, events):
		for trigger in triggers:
			self._trigger_refs[trigger] = self._trigger_refs.get(trigger, 0) + 1

		if self._started:
			self._schedule_action(triggers, callbacks, events)

		else:
			self._pending.append((triggers, callbacks, events))

	async def _loop_on_tasks(self):

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

		await self._loop_on_tasks()