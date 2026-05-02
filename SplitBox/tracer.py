import json
import threading
import time

TID_COMPUTE = 0
TID_COMM = 1

class Tracer:

    def __init__(self, worker_id=0):
        self.worker_id = worker_id
        self._events = []
        self._lock = threading.Lock()
        self._start = time.time()

    def reset(self, start=None):
        with self._lock:
            self._events = []
            self._start = start if start is not None else time.time()

    def record(self, name, tid, start, end, pid=None):
        with self._lock:
            self._events.append({
                "name": name,
                "ph": "X",
                "ts": (start - self._start) * 1e6,
                "dur": (end - start) * 1e6,
                "pid": pid if pid is not None else self.worker_id,
                "tid": tid,
            })

    def save(self, path=None):
        if path is None:
            path = f"worker_{self.worker_id}_trace.json"
        with self._lock:
            events = list(self._events)
        with open(path, "w") as f:
            json.dump({"traceEvents": events}, f)

    def get_events(self):
        with self._lock:
            return list(self._events)

    def get_traces(self):
        with self._lock:
            return {"traceEvents": list(self._events)}

    def get_time(self):
        return time.time()

    def wrap(self, name, coro, tid=TID_COMPUTE, pid=None):
        async def wrapped():
            t0 = time.time()
            result = await coro
            self.record(name, tid, t0, time.time(), pid=pid)
            return result
        return wrapped()

async def sync_clocks(client_tracer, tracers):
    t_ref = time.time()
    client_tracer.reset(t_ref)
    for tracer in tracers:
        t0 = time.time()
        t_worker = await tracer.get_time()
        t1 = time.time()
        delta = (t0 + t1) / 2 - t_worker
        await tracer.reset(t_ref - delta)
