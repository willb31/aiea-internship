"""
Correctness checks for the Double DQN implementation.

Run this before starting a long run. It takes well under a minute and catches
the mistakes that otherwise show up three hours into training as a flat learning
curve with no explanation.

    python test_ddqn.py

Everything from task 7 is carried over unchanged, since the replay buffer and
environment wrapper did not change. What is new is the Double DQN section, which
is worth reading: a wrong Double DQN does not crash, it just quietly trains a
slightly different algorithm and produces a plot that looks fine. So those tests
are built around the specific wrong versions that are easy to write --
evaluating with the online net, taking the bootstrap net's own max, swapping
which network selects and which evaluates -- and each one is rigged so the wrong
answer is a different number from the right one.

The replay buffer tests are pure NumPy and run anywhere. The rest need torch.
Tests that cannot run are reported as SKIP rather than silently passing.
"""

import sys
import traceback

import numpy as np

from env_and_replay import (
    DISCRETE_ACTIONS,
    ReplayBuffer,
    epsilon_at,
)

try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


PASS, FAIL, SKIP = [], [], []


def check(name, needs_torch=False):
    def deco(fn):
        if needs_torch and not HAVE_TORCH:
            SKIP.append(name)
            print(f"SKIP  {name}  (torch not importable)")
            return fn
        try:
            fn()
            PASS.append(name)
            print(f"ok    {name}")
        except Exception as e:
            FAIL.append((name, e))
            print(f"FAIL  {name}: {e}")
            traceback.print_exc()
        return fn
    return deco


# --------------------------------------------------------------------------
# Action space
# --------------------------------------------------------------------------
@check("discrete actions are inside CarRacing's bounds")
def _():
    assert DISCRETE_ACTIONS.shape == (5, 3), DISCRETE_ACTIONS.shape
    steer, gas, brake = DISCRETE_ACTIONS.T
    assert np.all(steer >= -1) and np.all(steer <= 1), "steering out of [-1, 1]"
    assert np.all(gas >= 0) and np.all(gas <= 1), "gas out of [0, 1]"
    assert np.all(brake >= 0) and np.all(brake <= 1), "brake out of [0, 1]"
    # A left and a right that are not the same action, and something that moves.
    assert not np.allclose(DISCRETE_ACTIONS[1], DISCRETE_ACTIONS[2])
    assert DISCRETE_ACTIONS[3][1] > 0, "no action applies throttle"


# --------------------------------------------------------------------------
# Epsilon schedule
# --------------------------------------------------------------------------
@check("epsilon anneals linearly then holds flat")
def _():
    assert epsilon_at(0, 1.0, 0.1, 1000) == 1.0
    assert abs(epsilon_at(500, 1.0, 0.1, 1000) - 0.55) < 1e-9
    assert abs(epsilon_at(1000, 1.0, 0.1, 1000) - 0.1) < 1e-9
    assert epsilon_at(50_000, 1.0, 0.1, 1000) == 0.1, "epsilon must not keep decaying"
    # monotone non-increasing
    vals = [epsilon_at(s, 1.0, 0.1, 1000) for s in range(0, 1200, 37)]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:])), "epsilon went up"


# --------------------------------------------------------------------------
# Replay buffer: the part most likely to be silently wrong
# --------------------------------------------------------------------------
def _buf(capacity=50, h=2, w=2, stack=4):
    return ReplayBuffer(capacity, img_stack=stack, height=h, width=w)


def _frame(v, h=2, w=2):
    return np.full((h, w), v, dtype=np.uint8)


@check("stacking within one episode returns the last 4 frames in order")
def _():
    b = _buf()
    for v in range(1, 9):
        b.add(_frame(v), action=0, reward=0.0, terminal=False, boundary=False)
    s = b._stack(7)  # absolute index 7 -> frame value 8
    got = [int(s[i][0, 0]) for i in range(4)]
    assert got == [5, 6, 7, 8], got


@check("stacking at the very start repeats the first frame instead of reading garbage")
def _():
    b = _buf()
    b.add(_frame(1), 0, 0.0, False, False)
    s = b._stack(0)
    got = [int(s[i][0, 0]) for i in range(4)]
    assert got == [1, 1, 1, 1], got

    b.add(_frame(2), 0, 0.0, False, False)
    s = b._stack(1)
    got = [int(s[i][0, 0]) for i in range(4)]
    assert got == [1, 1, 1, 2], got


@check("stacking never crosses an episode boundary")
def _():
    b = _buf()
    # episode A: frames 1..3, with 3 ending the episode
    for v in (1, 2, 3):
        b.add(_frame(v), 0, 0.0, terminal=(v == 3), boundary=(v == 3))
    # episode B: frames 10, 11
    b.add(_frame(10), 0, 0.0, False, False)
    b.add(_frame(11), 0, 0.0, False, False)

    s = b._stack(4)  # newest is frame 11, second-newest 10, then must stop
    got = [int(s[i][0, 0]) for i in range(4)]
    assert got == [10, 10, 10, 11], f"leaked across episode boundary: {got}"
    assert 3 not in got and 2 not in got, f"frames from previous episode leaked in: {got}"


@check("stacking respects boundaries produced by the stuck-timeout too")
def _():
    b = _buf()
    for v in (1, 2, 3, 4):
        # truncation: boundary True but terminal False
        b.add(_frame(v), 0, 0.0, terminal=False, boundary=(v == 4))
    b.add(_frame(20), 0, 0.0, False, False)
    s = b._stack(4)
    got = [int(s[i][0, 0]) for i in range(4)]
    assert got == [20, 20, 20, 20], got


@check("ring wraparound does not resurrect overwritten frames")
def _():
    cap = 10
    b = _buf(capacity=cap)
    for v in range(1, 26):  # 25 adds into a capacity-10 ring
        b.add(_frame(v % 256), 0, 0.0, False, False)
    assert len(b) == cap
    assert b._oldest == 15, b._oldest
    # newest absolute index is 24 (value 25)
    s = b._stack(24)
    got = [int(s[i][0, 0]) for i in range(4)]
    assert got == [22, 23, 24, 25], got
    # at the oldest valid index, history must be padded, not wrapped
    s = b._stack(15)
    got = [int(s[i][0, 0]) for i in range(4)]
    assert got == [16, 16, 16, 16], f"read overwritten data: {got}"


@check("sample returns matched (s, a, r, s') tuples with s' one step ahead")
def _():
    b = _buf(capacity=200)
    for v in range(1, 51):
        b.add(_frame(v), action=v % 5, reward=float(v), terminal=False, boundary=False)
    rng = np.random.default_rng(0)
    s, a, r, s2, term = b.sample(16, rng)
    assert s.shape == (16, 4, 2, 2), s.shape
    assert s2.shape == (16, 4, 2, 2), s2.shape
    assert a.shape == (16,) and r.shape == (16,) and term.shape == (16,)
    for k in range(16):
        newest = int(s[k][-1][0, 0])
        newest2 = int(s2[k][-1][0, 0])
        assert newest2 == newest + 1, f"next state is not one step ahead: {newest} -> {newest2}"
        # reward and action were stored alongside the state's newest frame
        assert float(r[k]) == float(newest), (r[k], newest)
        assert int(a[k]) == newest % 5, (a[k], newest)


@check("terminal flag cuts the bootstrap, truncation does not")
def _():
    b = _buf(capacity=50)
    b.add(_frame(1), 0, 0.0, terminal=True, boundary=True)    # real terminal
    b.add(_frame(2), 0, 0.0, terminal=False, boundary=True)   # stuck timeout
    b.add(_frame(3), 0, 0.0, terminal=False, boundary=False)
    b.add(_frame(4), 0, 0.0, terminal=False, boundary=False)
    assert bool(b.terminals[0]) is True
    assert bool(b.terminals[1]) is False, "a timeout must not be treated as terminal"
    assert bool(b.boundaries[1]) is True, "a timeout must still block stacking"


# --------------------------------------------------------------------------
# Bellman target
# --------------------------------------------------------------------------
@check("Q-learning target matches a hand computation", needs_torch=True)
def _():
    gamma = 0.9
    rewards = torch.tensor([1.0, 2.0, -3.0])
    next_q = torch.tensor([10.0, 5.0, 7.0])       # max_a' Q(s', a')
    terminals = torch.tensor([0.0, 1.0, 0.0])     # middle one is terminal

    y = rewards + gamma * next_q * (1.0 - terminals)

    # by hand:
    #   1.0 + 0.9 * 10.0 = 10.0
    #   2.0 + nothing (terminal) = 2.0
    #  -3.0 + 0.9 *  7.0 = 3.3
    expected = torch.tensor([10.0, 2.0, 3.3])
    assert torch.allclose(y, expected, atol=1e-6), (y, expected)


@check("gather picks Q for the action actually taken", needs_torch=True)
def _():
    q_all = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                          [9.0, 8.0, 7.0, 6.0, 5.0]])
    actions = torch.tensor([2, 0])
    q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
    assert torch.allclose(q, torch.tensor([3.0, 9.0])), q


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------
@check("network produces one Q value per action with the expected shapes", needs_torch=True)
def _():
    from ddqn_carracing import QNetwork
    net = QNetwork(img_stack=4, n_actions=5)
    x = torch.randint(0, 256, (7, 4, 96, 96), dtype=torch.uint8)
    out = net(x)
    assert out.shape == (7, 5), out.shape
    assert torch.isfinite(out).all(), "network produced NaN or inf"
    # The conv arithmetic in the docstring must actually hold. features() is the
    # bare conv stack, so it has to be fed the scaled float tensor that forward()
    # builds -- handing it raw uint8 is a mistake in the test, not the model.
    feat = net.features(x.float() / 128.0 - 1.0)
    assert feat.shape == (7, 32 * 10 * 10), feat.shape


@check("uint8 input is rescaled into roughly [-1, 1]", needs_torch=True)
def _():
    from ddqn_carracing import QNetwork
    net = QNetwork()
    x = torch.zeros((1, 4, 96, 96), dtype=torch.uint8)
    scaled = x.float() / 128.0 - 1.0
    assert abs(scaled.min().item() + 1.0) < 1e-6
    x = torch.full((1, 4, 96, 96), 255, dtype=torch.uint8)
    scaled = x.float() / 128.0 - 1.0
    assert 0.9 < scaled.max().item() < 1.0, scaled.max().item()


@check("the learner can actually drive loss down on a fixed batch", needs_torch=True)
def _():
    """
    The cheapest end-to-end check there is. If a network cannot overfit 8 fixed
    targets in 100 gradient steps, nothing about the training loop is worth
    running for three hours.
    """
    import torch.nn.functional as F
    from ddqn_carracing import QNetwork

    torch.manual_seed(0)
    net = QNetwork()
    opt = torch.optim.RMSprop(net.parameters(), lr=1e-3, alpha=0.95, eps=0.01)

    x = torch.randint(0, 256, (8, 4, 96, 96), dtype=torch.uint8)
    a = torch.randint(0, 5, (8,))
    y = torch.randn(8) * 5.0

    first = None
    for i in range(100):
        q = net(x).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
    last = loss.item()
    assert last < first * 0.25, f"loss barely moved: {first:.4f} -> {last:.4f}"


# --------------------------------------------------------------------------
# Double DQN. This is what is new in task 8, so it gets the most attention.
# --------------------------------------------------------------------------
class _FixedQ:
    """A stand-in network returning a preset Q table, so targets are checkable by hand."""

    def __init__(self, table):
        self.table = torch.tensor(table, dtype=torch.float32)

    def __call__(self, _states):
        return self.table


@check("double target values the online net's choice, not the target net's best", needs_torch=True)
def _():
    from ddqn_carracing import bellman_target

    # One transition. The two networks disagree about which action is best, and
    # that disagreement is the entire point of the method.
    #
    #   online says action 0 is best (value 5.0)
    #   bootstrap says action 1 is best (value 9.0), and rates action 0 at 1.0
    online = _FixedQ([[5.0, 4.0, 0.0]])
    boot = _FixedQ([[1.0, 9.0, 0.0]])

    r = torch.tensor([0.0])
    term = torch.tensor([0.0])

    # standard: takes the bootstrap net's own max -> 0 + 0.5 * 9.0 = 4.5
    y_std = bellman_target(online, boot, None, r, term, 0.5, double=False)
    assert torch.allclose(y_std, torch.tensor([4.5])), y_std

    # double: online picks action 0, bootstrap values it at 1.0 -> 0 + 0.5 * 1.0 = 0.5
    y_dbl = bellman_target(online, boot, None, r, term, 0.5, double=True)
    assert torch.allclose(y_dbl, torch.tensor([0.5])), y_dbl

    # The whole reason to do this: the double target is the smaller one whenever
    # the bootstrap net's argmax is inflated relative to the online net's.
    assert y_dbl.item() < y_std.item()


@check("double and standard targets agree when both nets are identical", needs_torch=True)
def _():
    """
    Sanity floor. If the two networks are the same, argmax from one and max from
    the other are the same number by definition. A double implementation that
    fails this is indexing something wrong.
    """
    from ddqn_carracing import bellman_target, QNetwork

    torch.manual_seed(0)
    net = QNetwork()
    same = QNetwork()
    same.load_state_dict(net.state_dict())

    s = torch.randint(0, 256, (16, 4, 96, 96), dtype=torch.uint8)
    r = torch.randn(16)
    term = (torch.rand(16) < 0.3).float()

    y_std = bellman_target(net, same, s, r, term, 0.99, double=False)
    y_dbl = bellman_target(net, same, s, r, term, 0.99, double=True)
    assert torch.allclose(y_std, y_dbl, atol=1e-5), (y_std - y_dbl).abs().max()


@check("double target is never above the standard target", needs_torch=True)
def _():
    """
    max_a Q_boot(s,a) >= Q_boot(s, argmax_a Q_online(s,a)) always, because the
    left side maximises over the same set the right side draws one element from.
    So the double target can only ever be less than or equal. If a random sweep
    ever produces a double target that is larger, selection and evaluation have
    been swapped.
    """
    from ddqn_carracing import bellman_target

    torch.manual_seed(1)
    for _ in range(200):
        online = _FixedQ(torch.randn(8, 5).tolist())
        boot = _FixedQ(torch.randn(8, 5).tolist())
        r = torch.randn(8)
        term = torch.zeros(8)
        y_std = bellman_target(online, boot, None, r, term, 0.99, double=False)
        y_dbl = bellman_target(online, boot, None, r, term, 0.99, double=True)
        assert (y_dbl <= y_std + 1e-6).all(), (y_std, y_dbl)


@check("terminal transitions drop the bootstrap under both targets", needs_torch=True)
def _():
    from ddqn_carracing import bellman_target

    # Row 1's bootstrap value under the online net's argmax (action 0) is 11.0,
    # deliberately not zero, so "non-terminal kept its bootstrap" is a real check
    # under the double target and not an accident of the table.
    online = _FixedQ([[100.0, 0.0], [100.0, 0.0]])
    boot = _FixedQ([[0.0, 500.0], [11.0, 500.0]])
    r = torch.tensor([7.0, 7.0])
    term = torch.tensor([1.0, 0.0])   # first is terminal, second is not

    for double in (False, True):
        y = bellman_target(online, boot, None, r, term, 0.99, double)
        assert abs(y[0].item() - 7.0) < 1e-6, f"terminal still bootstrapped ({double}): {y[0]}"
        assert y[1].item() != 7.0, "non-terminal lost its bootstrap"


@check("double target uses argmax, not max, from the online net", needs_torch=True)
def _():
    """
    A plausible wrong implementation is y = r + gamma * max_a Q_online(s',a'),
    dropping the bootstrap net entirely. Rig the numbers so that mistake gives a
    different answer from the correct one.
    """
    from ddqn_carracing import bellman_target

    online = _FixedQ([[3.0, 1.0]])    # online max is 3.0, argmax is action 0
    boot = _FixedQ([[20.0, 50.0]])    # bootstrap rates action 0 at 20.0

    y = bellman_target(online, boot, None, torch.tensor([0.0]), torch.tensor([0.0]), 1.0, True)
    assert abs(y.item() - 20.0) < 1e-6, f"expected 20.0 (Q_boot of online's argmax), got {y.item()}"
    assert abs(y.item() - 3.0) > 1e-6, "used the online net's value instead of the bootstrap net's"
    assert abs(y.item() - 50.0) > 1e-6, "used the bootstrap net's own max, that is the standard target"


@check("--double is wired to the parser and defaults off", needs_torch=True)
def _():
    from ddqn_carracing import build_parser

    a = build_parser().parse_args([])
    assert a.double is False, "double must be opt-in so the baseline run is unchanged"
    a = build_parser().parse_args(["--double", "--target-net"])
    assert a.double is True and a.target_net is True
    # Task 8 caps by steps only. The flag still exists for resuming, but nothing
    # should be capping on wall clock by default.
    assert a.max_hours is None, "a default wall-clock cap would repeat task 7's mistake"


# --------------------------------------------------------------------------
# Environment, slowest test so it goes last
# --------------------------------------------------------------------------
@check("the wrapper resets and steps with the right shapes and dtypes")
def _():
    from env_and_replay import CarRacingDiscrete
    env = CarRacingDiscrete(seed=0, action_repeat=2)
    frame, state = env.reset(seed=0)
    assert frame.shape == (96, 96) and frame.dtype == np.uint8, (frame.shape, frame.dtype)
    assert state.shape == (4, 96, 96) and state.dtype == np.uint8, (state.shape, state.dtype)
    # a fresh stack is 4 copies of the same frame
    assert np.array_equal(state[0], state[3])

    for a in range(5):
        nf, ns, r, term, trunc = env.step(a)
        assert nf.shape == (96, 96) and ns.shape == (4, 96, 96)
        assert isinstance(float(r), float)
        assert isinstance(bool(term), bool) and isinstance(bool(trunc), bool)
    # after stepping, the stack has actually shifted
    assert not np.array_equal(ns[0], ns[3]), "frame stack is not updating"
    env.close()


# --------------------------------------------------------------------------
if __name__ == "__main__":
    print()
    print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    if SKIP:
        print("skipped:", ", ".join(SKIP))
    if FAIL:
        print("\nFAILURES:")
        for name, e in FAIL:
            print(f"  {name}: {e}")
        sys.exit(1)
    print("\nAll good. Safe to start a long run.")
