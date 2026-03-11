import asyncio
import pytest
from SplitBox.events import EventEmitter


@pytest.fixture
def emitter():
    return EventEmitter()


# --- sync listeners ---

def test_on_and_emit_sync(emitter):
    results = []
    emitter.on("data", lambda x: results.append(x))
    emitter.emit("data", 42)
    assert results == [42]


def test_emit_returns_false_with_no_listeners(emitter):
    assert emitter.emit("nothing") is False


def test_emit_returns_true_with_listeners(emitter):
    emitter.on("ping", lambda: None)
    assert emitter.emit("ping") is True


def test_multiple_listeners(emitter):
    results = []
    emitter.on("x", lambda: results.append(1))
    emitter.on("x", lambda: results.append(2))
    emitter.emit("x")
    assert results == [1, 2]


def test_off_removes_listener(emitter):
    results = []
    def handler():
        results.append(1)
    emitter.on("x", handler)
    emitter.off("x", handler)
    emitter.emit("x")
    assert results == []


def test_once_fires_only_once(emitter):
    results = []
    emitter.once("x", lambda: results.append(1))
    emitter.emit("x")
    emitter.emit("x")
    assert results == [1]


def test_listener_count(emitter):
    assert emitter.listener_count("x") == 0
    emitter.on("x", lambda: None)
    emitter.on("x", lambda: None)
    assert emitter.listener_count("x") == 2


def test_emit_passes_args_and_kwargs(emitter):
    received = {}
    def handler(a, b, key=None):
        received["a"] = a
        received["b"] = b
        received["key"] = key
    emitter.on("msg", handler)
    emitter.emit("msg", 1, 2, key="val")
    assert received == {"a": 1, "b": 2, "key": "val"}


# --- async listeners ---

@pytest.mark.asyncio
async def test_async_listener_fires(emitter):
    results = []
    async def handler(x):
        results.append(x)
    emitter.on("data", handler)
    emitter.emit("data", 99)
    await asyncio.sleep(0)  # yield control so the task can run
    assert results == [99]


@pytest.mark.asyncio
async def test_emit_does_not_block(emitter):
    order = []
    async def slow_handler():
        await asyncio.sleep(0.05)
        order.append("handler")
    emitter.on("go", slow_handler)
    emitter.emit("go")
    order.append("after emit")
    await asyncio.sleep(0.1)
    assert order == ["after emit", "handler"]


@pytest.mark.asyncio
async def test_once_with_async_listener(emitter):
    results = []
    async def handler():
        results.append(1)
    emitter.once("x", handler)
    emitter.emit("x")
    emitter.emit("x")
    await asyncio.sleep(0)
    assert results == [1]


@pytest.mark.asyncio
async def test_multiple_async_listeners_all_fire(emitter):
    results = []
    async def h1():
        results.append("a")
    async def h2():
        results.append("b")
    emitter.on("go", h1)
    emitter.on("go", h2)
    emitter.emit("go")
    await asyncio.sleep(0)
    assert sorted(results) == ["a", "b"]
