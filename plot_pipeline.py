import asyncio
import random
import time

from matplotlib import pyplot as plt

from pipeline_parallel import get_pipeline_parallel_flow, parse_task_str

task_timings = []
global_start = time.time()

async def fake_task(task_str, t):

	start = time.time() - global_start

	print(f'starting {task_str} for {t} seconds')
	await asyncio.sleep(t)
	print(f'{task_str} ended')

	end = time.time() - global_start

	data_point = {"task_str": task_str, "start": start, "end": end}
	task_timings.append(data_point)

def plot_timings():

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

	plt.bar(f_x, f_tops, bottom=f_bottoms, edgecolor="black", color="red")
	plt.bar(b_x, b_tops, bottom=b_bottoms, edgecolor="black", color="blue")

	plt.show()

def metrics_wrapper(task_str, coro):

    async def wrapped():
        start = time.time() - global_start

        # print(f'starting {task_str}')
        result = await coro
        # print(f'{task_str} ended')

        end = time.time() - global_start

        data_point = {"task_str": task_str, "start": start, "end": end}
        task_timings.append(data_point)

        return result

    return wrapped()

async def main():

	num_workers = 3
	num_stages = 15

	delta = lambda : random.random()/50+0.1
	# delta = lambda : 0.1
	def get_pipeline_stages(j, x):
		pipeline_stages = []
		for i in range(num_workers): pipeline_stages.append(fake_task(task_str=f'f{i+1}s{j+1}', t=0.5*delta()))
		for i in range(num_workers): pipeline_stages.append(fake_task(task_str=f'b{num_workers-i}s{j+1}', t=1*delta()))

		return pipeline_stages

	batch = list(i+1 for i in range(num_stages))

	flow = get_pipeline_parallel_flow(num_workers, get_pipeline_stages, batch)

	await flow.start()

	plot_timings()

if (__name__ == '__main__'):
	asyncio.run(main())