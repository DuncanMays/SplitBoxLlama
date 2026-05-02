import asyncio
import time

from SplitBox.tracer import Tracer, TID_COMPUTE, sync_clocks
from SplitBox.tests.multi_stub_test import async_wrapper

def test_sync_clocks_aligns_epochs():
	async def _test():
		client_tracer = Tracer(worker_id=0)
		worker_tracers = [Tracer(worker_id=i + 1) for i in range(3)]
		wrapped = [async_wrapper(t) for t in worker_tracers]

		await sync_clocks(client_tracer, wrapped)

		now = time.time()
		for wt in worker_tracers:
			wt.record("test", TID_COMPUTE, now, now + 0.001)
		client_tracer.record("test", TID_COMPUTE, now, now + 0.001)

		client_ts = client_tracer.get_events()[0]["ts"]
		for wt in worker_tracers:
			worker_ts = wt.get_events()[0]["ts"]
			assert abs(worker_ts - client_ts) < 5000  # within 5ms

	asyncio.run(_test())
