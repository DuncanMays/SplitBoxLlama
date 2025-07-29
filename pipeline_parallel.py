import asyncio
import random
import time

from matplotlib import pyplot as plt

from events import EventFlow, print_msg

task_timings = []
global_start = time.time()

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

async def fake_task(task_str, t):

	start = time.time() - global_start

	print(f'starting {task_str} for {t} seconds')
	await asyncio.sleep(t)
	print(f'{task_str} ended')

	end = time.time() - global_start

	data_point = {"task_str": task_str, "start": start, "end": end}
	task_timings.append(data_point)

def plot_timings():
	print("plotting!")

	f_x = []
	f_tops = []
	f_bottoms = []
	b_x = []
	b_tops = []
	b_bottoms = []

	for dp in task_timings:
		d, w, s = parse_task_str(dp["task_str"])

		if (d=="f"):
			f_x.append(w)
			f_tops.append(dp["end"] - dp["start"])
			f_bottoms.append(dp["start"])

		if (d=="b"):
			b_x.append(w)
			b_tops.append(dp["end"] - dp["start"])
			b_bottoms.append(dp["start"])

	plt.bar(f_x, f_tops, bottom=f_bottoms, edgecolor="black", color="blue")
	plt.bar(b_x, b_tops, bottom=b_bottoms, edgecolor="black", color="red")

	plt.show()

def get_pipeline_parallel_flow(num_workers, get_pipeline_stages, batch):
	
	pipeline_prefixes = []
	for i in range(num_workers): pipeline_prefixes.append(f'f{i+1}')
	for i in range(num_workers): pipeline_prefixes.append(f'b{num_workers-i}')

	# tasks are indicated with a letter and two numbers
	# the letter, either f or b means forward or backwards
	# the first number indicates the worker
	# the second number indicates the batch

	flow = EventFlow()

	for minibatch_index in range(len(batch)):

		pipeline_stages = get_pipeline_stages(minibatch_index)

		if ((len(pipeline_stages) // num_workers) != 2):
			raise BaseException(f"Incompatible number of pipeline stages and workers:{len(pipeline_stages)}, {num_workers}")

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


async def main():

	num_workers = 3

	# delta = lambda : random.random()/50+0.1
	delta = lambda : 0.1
	def get_pipeline_stages(j):
		pipeline_stages = []
		for i in range(num_workers): pipeline_stages.append(fake_task(task_str=f'f{i+1}s{j+1}', t=0.5*delta()))
		for i in range(num_workers): pipeline_stages.append(fake_task(task_str=f'b{num_workers-i}s{j+1}', t=1*delta()))

		return pipeline_stages

	batch = [1,2,3,4,5]

	flow = get_pipeline_parallel_flow(num_workers, get_pipeline_stages, batch)

	await flow.start()

	plot_timings()

if (__name__ == '__main__'):
	asyncio.run(main())