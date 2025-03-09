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


async def main():

	num_workers = 3
	num_stages = 15
	pipeline_prefixes = ['f1', 'f2', 'f3', 'b3', 'b2', 'b1']

	# tasks are indicated with a letter and two numbers
	# the letter, either f or b means forward or backwards
	# the first number indicates the worker
	# the second number indicates the batch

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

			print(f'triggers for {task_str} are {triggers}')

			# callbacks = [print_msg(msg=task_str, t=1)]
			callbacks = [print_msg(msg=task_str, t=0.5+random.random())]

			events = [task_str]

			flow.set_action(triggers, callbacks, events)

	await flow.start()

if (__name__ == '__main__'):
	asyncio.run(main())