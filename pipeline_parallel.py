import asyncio
import random
from events import EventFlow, print_msg

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

def get_pipeline_parallel_flow(worker_stubs, num_stages, forward_args, backward_args):

	num_workers = len(worker_stubs)

	param_list = forward_args + backward_args
	pipeline_prefixes = []
	ordered_stubs = []

	for i in range(0, len(worker_stubs), 1):
		pipeline_prefixes.append(f'f{i+1}')
		ordered_stubs.append(worker_stubs[i].forward)

	for i in range(len(worker_stubs), 0, -1):
		pipeline_prefixes.append(f'b{i}')
		ordered_stubs.append(worker_stubs[i-1].backward)

	flow = EventFlow()

	for s in range(1, num_stages+1):

		for i, prefix in enumerate(pipeline_prefixes):

			task_str = f'{prefix}s{s}'

			triggers = []

			# if not the first stage, add the prior task on the worker as a trigger
			if (s != 1):
				triggers.append(get_prior_task(task_str, num_workers, num_stages))

			# if not the first pipeline step, add the prior pipeline step as a trigger
			if (i != 0):
				prior_prefix = pipeline_prefixes[i-1]
				triggers.append(f'{prior_prefix}s{s}')

			# print(f'triggers for {task_str} are {triggers}')

			stub = ordered_stubs[i]
			args = param_list[i]

			callbacks = [stub(args)]

			# callbacks = [print_msg(msg=task_str, t=1)]
			# callbacks = [print_msg(msg=task_str, t=0.5+random.random())]

			events = [task_str]

			flow.set_action(triggers, callbacks, events)

	return flow

async def main():

	class MockStub():

		def __init__(self, msg, t=1):
			self.msg = msg
			self.t = t

		async def forward(self, s):
			print(f'forwards on {self.msg} for {self.t} seconds, {s}')
			await asyncio.sleep(self.t)
			print(f'forwards on {self.msg} ended')

		async def backward(self, s):
			print(f'backwards on {self.msg} for {self.t} seconds, {s}')
			await asyncio.sleep(self.t)
			print(f'backwards on {self.msg} ended')

	worker_ids = ['worker 1', 'worker 2', 'worker 3', 'worker 4']
	worker_stubs = [MockStub(i) for i in worker_ids]

	forward_args = ['first forward', 'second forward', 'third forward', 'fourth forward']
	backward_args = ['first backward', 'second backward', 'third backward', 'fourth backward']

	flow = get_pipeline_parallel_flow(worker_stubs, 10, forward_args, backward_args)

	await flow.start()
	

if (__name__ == '__main__'):
	asyncio.run(main())