import asyncio

async def print_msg(msg='hello!', t=1):
	print(f'waiting {t} seconds')
	await asyncio.sleep(t)
	print(msg)

class EventFlow():

	def __init__(self):
		self.event_dict = {}
		self.actions = []
		self.start_event = asyncio.Event()

	async def set_action_helper(self, triggers, callbacks, events):

		if (len(triggers) == 0):
			triggers = [self.start_event]

		T = [t.wait() for t in triggers]
		await asyncio.gather(*T)

		await asyncio.gather(*callbacks)

		for e in events:
			e.set()

	def get_event(self, event_name):
		e = None

		if event_name in self.event_dict:
			e = self.event_dict[event_name]

		else:
			e = asyncio.Event()
			self.event_dict[event_name] = e

		return e

	def set_action(self, triggers, callbacks, events):

		triggers = [self.get_event(t) for t in triggers]
		events = [self.get_event(e) for e in events]

		self.actions.append(self.set_action_helper(triggers, callbacks, events))

	async def start(self):
		self.start_event.set()
		await asyncio.gather(*self.actions)

async def main():

	flow = EventFlow()

	flow.set_action([], [print_msg(msg='start', t=0.1)], ['e1'])
	flow.set_action(['e1'], [print_msg(msg='first stage', t=1.1)], ['e2', 'e3'])
	flow.set_action(['e2'], [print_msg(msg='second stage a', t=1), print_msg(msg='first stage b', t=1.5)], ['e4'])
	flow.set_action(['e3', 'e4'], [print_msg(msg='third stage', t=0.8)], ['e5'])

	print('start_flow')
	await flow.start()
	print('flow completed')

if (__name__ == '__main__'):
	asyncio.run(main())