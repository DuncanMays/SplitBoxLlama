import pytest
import asyncio
from SplitBox.pipeline_parallel import EventFlow


def cb(log, label):
    async def _cb():
        log.append(label)
    return _cb()


# --- basic execution ---

@pytest.mark.asyncio
async def test_empty_flow():
    await EventFlow().start()


@pytest.mark.asyncio
async def test_no_triggers_runs_immediately():
    ran = []
    flow = EventFlow()
    flow.set_action([], [cb(ran, 'a')], [])
    await flow.start()
    assert ran == ['a']


@pytest.mark.asyncio
async def test_multiple_no_trigger_actions_all_run():
    ran = []
    flow = EventFlow()
    for label in ['x', 'y', 'z']:
        flow.set_action([], [cb(ran, label)], [])
    await flow.start()
    assert set(ran) == {'x', 'y', 'z'}


# --- single trigger chains ---

@pytest.mark.asyncio
async def test_single_trigger_chain():
    ran = []
    flow = EventFlow()
    flow.set_action([], [cb(ran, 'first')], ['done'])
    flow.set_action(['done'], [cb(ran, 'second')], [])
    await flow.start()
    assert ran == ['first', 'second']


@pytest.mark.asyncio
async def test_chain_of_three():
    ran = []
    flow = EventFlow()
    flow.set_action([], [cb(ran, 'A')], ['a_done'])
    flow.set_action(['a_done'], [cb(ran, 'B')], ['b_done'])
    flow.set_action(['b_done'], [cb(ran, 'C')], [])
    await flow.start()
    assert ran == ['A', 'B', 'C']


# --- multiple triggers ---

@pytest.mark.asyncio
async def test_multiple_triggers_all_must_fire():
    ran = []
    flow = EventFlow()
    flow.set_action([], [cb(ran, 'left')], ['left_done'])
    flow.set_action([], [cb(ran, 'right')], ['right_done'])
    flow.set_action(['left_done', 'right_done'], [cb(ran, 'joined')], [])
    await flow.start()
    assert 'left' in ran
    assert 'right' in ran
    assert ran[-1] == 'joined'


@pytest.mark.asyncio
async def test_diamond_dependency():
    ran = []
    flow = EventFlow()
    flow.set_action([], [cb(ran, 'A')], ['a_done'])
    flow.set_action(['a_done'], [cb(ran, 'B')], ['b_done'])
    flow.set_action(['a_done'], [cb(ran, 'C')], ['c_done'])
    flow.set_action(['b_done', 'c_done'], [cb(ran, 'D')], [])
    await flow.start()
    assert ran[0] == 'A'
    assert set(ran[1:3]) == {'B', 'C'}
    assert ran[-1] == 'D'


# --- fan-out and fan-in ---

@pytest.mark.asyncio
async def test_one_event_triggers_multiple_actions():
    ran = []
    flow = EventFlow()
    flow.set_action([], [cb(ran, 'source')], ['ready'])
    flow.set_action(['ready'], [cb(ran, 'branch_a')], [])
    flow.set_action(['ready'], [cb(ran, 'branch_b')], [])
    await flow.start()
    assert ran[0] == 'source'
    assert set(ran[1:]) == {'branch_a', 'branch_b'}


@pytest.mark.asyncio
async def test_action_fires_multiple_events():
    ran = []
    flow = EventFlow()
    flow.set_action([], [cb(ran, 'source')], ['ev_a', 'ev_b'])
    flow.set_action(['ev_a'], [cb(ran, 'needs_a')], [])
    flow.set_action(['ev_b'], [cb(ran, 'needs_b')], [])
    await flow.start()
    assert ran[0] == 'source'
    assert set(ran[1:]) == {'needs_a', 'needs_b'}


# --- dynamic registration ---

@pytest.mark.asyncio
async def test_callback_adds_new_action():
    ran = []
    flow = EventFlow()

    async def first():
        ran.append('first')
        flow.set_action([], [cb(ran, 'added')], [])

    flow.set_action([], [first()], [])
    await flow.start()
    assert ran == ['first', 'added']


@pytest.mark.asyncio
async def test_callback_adds_dependent_chain():
    ran = []
    flow = EventFlow()

    async def seed():
        ran.append('seed')
        flow.set_action([], [cb(ran, 'step1')], ['step1_done'])
        flow.set_action(['step1_done'], [cb(ran, 'step2')], [])

    flow.set_action([], [seed()], [])
    await flow.start()
    assert ran == ['seed', 'step1', 'step2']


@pytest.mark.asyncio
async def test_late_action_on_already_fired_event():
    ran = []
    flow = EventFlow()
    flow.set_action([], [cb(ran, 'producer')], ['ready'])
    await flow.start()

    flow.set_action(['ready'], [cb(ran, 'late_consumer')], [])
    await flow._join()

    assert ran == ['producer', 'late_consumer']


@pytest.mark.asyncio
async def test_tasks_empty_after_completion():
    flow = EventFlow()
    flow.set_action([], [cb([], 'a')], ['done'])
    flow.set_action(['done'], [cb([], 'b')], [])
    await flow.start()
    assert len(flow._tasks) == 0
