import pytest
import asyncio

from SplitBox.pipeline_parallel import EventFlow


# --- helpers ---

def make_ordered_callback(order_log, label):
    async def cb():
        order_log.append(label)
    return cb()


# --- tests ---

@pytest.mark.asyncio
async def test_single_action_no_triggers():
    """An action with no triggers runs immediately on start."""
    ran = []

    flow = EventFlow()
    flow.set_action([], [make_ordered_callback(ran, 'a')], [])

    await flow.start()

    assert ran == ['a']


@pytest.mark.asyncio
async def test_single_trigger():
    """An action blocked on one trigger runs after that trigger fires."""
    ran = []

    flow = EventFlow()
    flow.set_action([], [make_ordered_callback(ran, 'first')], ['done_first'])
    flow.set_action(['done_first'], [make_ordered_callback(ran, 'second')], [])

    await flow.start()

    assert ran == ['first', 'second']


@pytest.mark.asyncio
async def test_chain_of_three():
    """A → B → C executes in strict order."""
    ran = []

    flow = EventFlow()
    flow.set_action([], [make_ordered_callback(ran, 'A')], ['a_done'])
    flow.set_action(['a_done'], [make_ordered_callback(ran, 'B')], ['b_done'])
    flow.set_action(['b_done'], [make_ordered_callback(ran, 'C')], [])

    await flow.start()

    assert ran == ['A', 'B', 'C']


@pytest.mark.asyncio
async def test_multiple_triggers_all_must_fire():
    """An action with two triggers only runs after both have fired."""
    ran = []

    flow = EventFlow()
    flow.set_action([], [make_ordered_callback(ran, 'left')], ['left_done'])
    flow.set_action([], [make_ordered_callback(ran, 'right')], ['right_done'])
    flow.set_action(
        ['left_done', 'right_done'],
        [make_ordered_callback(ran, 'joined')],
        []
    )

    await flow.start()

    assert 'left' in ran
    assert 'right' in ran
    assert ran[-1] == 'joined'


@pytest.mark.asyncio
async def test_one_event_triggers_multiple_actions():
    """A single fired event can unblock multiple downstream actions."""
    ran = []

    flow = EventFlow()
    flow.set_action([], [make_ordered_callback(ran, 'source')], ['source_done'])
    flow.set_action(['source_done'], [make_ordered_callback(ran, 'branch_a')], [])
    flow.set_action(['source_done'], [make_ordered_callback(ran, 'branch_b')], [])

    await flow.start()

    assert ran[0] == 'source'
    assert set(ran[1:]) == {'branch_a', 'branch_b'}


@pytest.mark.asyncio
async def test_independent_actions_all_run():
    """Multiple actions with no triggers all run."""
    ran = []

    flow = EventFlow()
    for label in ['x', 'y', 'z']:
        flow.set_action([], [make_ordered_callback(ran, label)], [])

    await flow.start()

    assert set(ran) == {'x', 'y', 'z'}


@pytest.mark.asyncio
async def test_action_fires_multiple_events():
    """An action that fires two events can unblock two separate downstream actions."""
    ran = []

    flow = EventFlow()
    flow.set_action([], [make_ordered_callback(ran, 'source')], ['ev_a', 'ev_b'])
    flow.set_action(['ev_a'], [make_ordered_callback(ran, 'needs_a')], [])
    flow.set_action(['ev_b'], [make_ordered_callback(ran, 'needs_b')], [])

    await flow.start()

    assert ran[0] == 'source'
    assert set(ran[1:]) == {'needs_a', 'needs_b'}


@pytest.mark.asyncio
async def test_diamond_dependency():
    """A → (B, C) → D: D waits for both B and C, which both wait for A."""
    ran = []

    flow = EventFlow()
    flow.set_action([], [make_ordered_callback(ran, 'A')], ['a_done'])
    flow.set_action(['a_done'], [make_ordered_callback(ran, 'B')], ['b_done'])
    flow.set_action(['a_done'], [make_ordered_callback(ran, 'C')], ['c_done'])
    flow.set_action(
        ['b_done', 'c_done'],
        [make_ordered_callback(ran, 'D')],
        []
    )

    await flow.start()

    assert ran[0] == 'A'
    assert set(ran[1:3]) == {'B', 'C'}
    assert ran[-1] == 'D'


@pytest.mark.asyncio
async def test_callbacks_receive_correct_side_effects():
    """Callbacks can mutate shared state and that state is visible after start()."""
    counter = {'value': 0}

    async def increment():
        counter['value'] += 1

    flow = EventFlow()
    flow.set_action([], [increment()], ['done'])
    flow.set_action(['done'], [increment()], [])

    await flow.start()

    assert counter['value'] == 2


@pytest.mark.asyncio
async def test_get_event_is_idempotent():
    """Calling get_event with the same name twice returns the same object."""
    flow = EventFlow()

    e1 = flow.get_event('my_event')
    e2 = flow.get_event('my_event')

    assert e1 is e2


@pytest.mark.asyncio
async def test_empty_flow_starts_cleanly():
    """A flow with no actions completes without error."""
    flow = EventFlow()
    await flow.start()
