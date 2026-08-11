"""
Correctness checks for the task 9 benchmarking framework.

    python test_framework.py

Run this before starting a benchmark. It takes under a minute.

The DQN target tests are carried over from task 8 because that code did not
change. What is new here is the framework surface, and the tests are aimed at
the failures that a benchmark makes uniquely dangerous. A benchmark that runs
the same algorithm twice under two names, or that gives one algorithm a bigger
step budget than another, does not crash. It produces a clean-looking table of
numbers that are wrong, and every conclusion drawn from it is wrong too.

So the registry tests check that the five entries really are five different
algorithms, and the runner tests check that the step accounting is identical
across agents that collect experience at different rates.
"""

import sys
import traceback

import numpy as np
import torch

import rl_benchmark as R

PASS, FAIL = [], []


def check(name):
    def deco(fn):
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


class StubEnv:
    def __init__(self, seed=0, img_stack=4, action_repeat=4, render_mode=None):
        self.img_stack = img_stack
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def _obs(self):
        f = self.rng.integers(0, 256, (96, 96), dtype=np.uint8)
        return f, np.repeat(f[None], self.img_stack, axis=0)

    def reset(self, seed=None):
        self.t = 0
        return self._obs()

    def step(self, a):
        self.t += 1
        f, s = self._obs()
        return f, s, 1.0 if a == 3 else -0.1, self.t >= 40, False

    def close(self):
        pass


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
@check("every registered algorithm builds a config and an agent")
def _():
    dev = torch.device("cpu")
    for name in R.ALGORITHMS:
        cfg = R.build_config(name, seed=0)
        agent = R.build_agent(name, cfg, dev)
        assert isinstance(agent, R.Agent), name
        assert agent.num_envs >= 1, name


@check("the five registry entries are five genuinely different algorithms")
def _():
    """
    The failure this guards against is a copy-paste in the registry that leaves
    two entries with identical flags. Both would train, both would appear in the
    table, and the write-up would report a difference between them that is pure
    seed noise.
    """
    sigs = {}
    for name, spec in R.ALGORITHMS.items():
        cfg = R.build_config(name)
        sig = (spec["agent"], cfg.target_net, cfg.double, cfg.dueling)
        assert sig not in sigs, f"{name} is identical to {sigs[sig]}: {sig}"
        sigs[sig] = name
    assert len(sigs) == len(R.ALGORITHMS)


@check("registry labels and colours are unique")
def _():
    labels = [s["label"] for s in R.ALGORITHMS.values()]
    colors = [s["color"] for s in R.ALGORITHMS.values()]
    assert len(set(labels)) == len(labels), "duplicate label would mislabel a curve"
    assert len(set(colors)) == len(colors), "duplicate colour would merge two curves"


@check("Config rejects unknown keys instead of silently ignoring them")
def _():
    """A typo'd override that is silently dropped means running the default
    config while believing you ran something else."""
    try:
        R.Config(learning_rate=1e-4)   # correct name is lr
    except KeyError:
        return
    raise AssertionError("Config accepted an unknown key")


@check("the dqn variants differ only in the three documented flags")
def _():
    base = R.build_config("dqn2013").as_dict()
    for name in ("dqn2015", "ddqn", "dueling_ddqn"):
        d = R.build_config(name).as_dict()
        differing = {k for k in base if base[k] != d[k]}
        assert differing <= {"target_net", "double", "dueling"}, (name, differing)


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------
@check("both Q networks give one value per action with the right shape")
def _():
    x = torch.randint(0, 256, (7, 4, 96, 96), dtype=torch.uint8)
    for net in (R.QNetwork(), R.DuelingQNetwork()):
        out = net(x)
        assert out.shape == (7, 5), (type(net).__name__, out.shape)
        assert torch.isfinite(out).all(), type(net).__name__


@check("dueling advantages are mean-zero, so Q averages to V")
def _():
    """
    Q = V + A - mean(A) means the mean of Q over actions is exactly V. If that
    identity fails, the two streams are not being combined the way the paper
    specifies and the decomposition is unidentifiable.
    """
    torch.manual_seed(0)
    net = R.DuelingQNetwork()
    x = torch.randint(0, 256, (5, 4, 96, 96), dtype=torch.uint8)
    with torch.no_grad():
        q = net(x)
        h = net.features(x.float() / 128.0 - 1.0)
        v = net.value(h).squeeze(1)
    assert torch.allclose(q.mean(dim=1), v, atol=1e-5), (q.mean(dim=1) - v).abs().max()


@check("the actor critic returns a valid distribution and a scalar value")
def _():
    net = R.ActorCritic()
    x = torch.randint(0, 256, (6, 4, 96, 96), dtype=torch.uint8)
    logits, value = net(x)
    assert logits.shape == (6, 5) and value.shape == (6,), (logits.shape, value.shape)
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(6), atol=1e-5)
    # near-uniform at init, so the first rollouts explore
    assert probs.max().item() < 0.5, f"policy starts too confident: {probs.max().item():.3f}"


# --------------------------------------------------------------------------
# Targets, carried over from task 8
# --------------------------------------------------------------------------
class _FixedQ:
    def __init__(self, table):
        self.table = torch.tensor(table, dtype=torch.float32)

    def __call__(self, _):
        return self.table


@check("double target values the online net's choice, not the target net's best")
def _():
    online = _FixedQ([[5.0, 4.0, 0.0]])
    boot = _FixedQ([[1.0, 9.0, 0.0]])
    r, term = torch.tensor([0.0]), torch.tensor([0.0])
    y_std = R.bellman_target(online, boot, None, r, term, 0.5, False)
    y_dbl = R.bellman_target(online, boot, None, r, term, 0.5, True)
    assert torch.allclose(y_std, torch.tensor([4.5])), y_std
    assert torch.allclose(y_dbl, torch.tensor([0.5])), y_dbl


@check("double target is never above the standard target")
def _():
    torch.manual_seed(1)
    for _ in range(100):
        online = _FixedQ(torch.randn(8, 5).tolist())
        boot = _FixedQ(torch.randn(8, 5).tolist())
        r, term = torch.randn(8), torch.zeros(8)
        assert (R.bellman_target(online, boot, None, r, term, 0.99, True)
                <= R.bellman_target(online, boot, None, r, term, 0.99, False) + 1e-6).all()


@check("terminal transitions drop the bootstrap, truncations keep it")
def _():
    online = _FixedQ([[100.0, 0.0], [100.0, 0.0]])
    boot = _FixedQ([[0.0, 500.0], [11.0, 500.0]])
    r = torch.tensor([7.0, 7.0])
    term = torch.tensor([1.0, 0.0])
    for double in (False, True):
        y = R.bellman_target(online, boot, None, r, term, 0.99, double)
        assert abs(y[0].item() - 7.0) < 1e-6, f"terminal bootstrapped ({double})"
        assert abs(y[1].item() - 7.0) > 1e-6, f"non-terminal lost bootstrap ({double})"


# --------------------------------------------------------------------------
# GAE
# --------------------------------------------------------------------------
@check("GAE matches a hand computation")
def _():
    # one env, three steps, no terminals. gamma = 1, lambda = 1 makes the
    # advantage the plain discounted return minus the baseline.
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.tensor([[0.0], [0.0], [0.0]])
    dones = torch.tensor([[0.0], [0.0], [0.0]])
    last = torch.tensor([0.0])
    adv = R.compute_gae(rewards, values, dones, last, gamma=1.0, lam=1.0)
    # working backwards: t2 = 1, t1 = 1 + 1 = 2, t0 = 1 + 2 = 3
    assert torch.allclose(adv, torch.tensor([[3.0], [2.0], [1.0]])), adv


@check("GAE cuts the chain at a real terminal")
def _():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.tensor([[0.0], [0.0], [0.0]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])   # episode ends at t=1
    last = torch.tensor([0.0])
    adv = R.compute_gae(rewards, values, dones, last, gamma=1.0, lam=1.0)
    # t1 is terminal so it gets its own reward only, and t0 must not see past it
    assert abs(adv[1].item() - 1.0) < 1e-6, adv
    assert abs(adv[0].item() - 2.0) < 1e-6, adv


# --------------------------------------------------------------------------
# Agent interface
# --------------------------------------------------------------------------
@check("policy_probs is a probability distribution for every agent")
def _():
    dev = torch.device("cpu")
    states = np.random.randint(0, 256, (16, 4, 96, 96), dtype=np.uint8)
    for name in R.ALGORITHMS:
        agent = R.build_agent(name, R.build_config(name), dev)
        p = agent.policy_probs(states)
        assert p.shape == (5,), (name, p.shape)
        assert abs(p.sum() - 1.0) < 1e-4, (name, p.sum())
        assert (p >= -1e-9).all(), (name, p)


@check("every agent honours the step budget identically")
def _():
    """
    The one that matters most for a benchmark. PPO collects 4 transitions per
    runner iteration and DQN collects 1, so a runner that counted iterations
    would hand PPO four times the data and the whole comparison would be void.
    """
    R.CarRacingDiscrete = StubEnv
    budget = 400
    for name in R.ALGORITHMS:
        extra = (dict(rollout=8, batch_size=16, epochs=1)
                 if R.ALGORITHMS[name]["agent"] == "ppo"
                 else dict(learn_start=50, buffer_size=500))
        s = R.run_one(name, "/tmp/_tf_budget", steps=budget, device="cpu",
                      quiet=True, **extra)
        assert budget <= s["steps"] < budget + 8, (name, s["steps"])


@check("a run writes all three logs and a summary")
def _():
    import os
    R.CarRacingDiscrete = StubEnv
    R.run_one("ddqn", "/tmp/_tf_logs", steps=400, device="cpu", quiet=True,
              learn_start=50, buffer_size=500, log_every=50, policy_every=100)
    for suffix in ("episodes.csv", "updates.csv", "policy.csv", "summary.json"):
        p = f"/tmp/_tf_logs/ddqn_{suffix}"
        assert os.path.exists(p) and os.path.getsize(p) > 0, p


# --------------------------------------------------------------------------
if __name__ == "__main__":
    print()
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("\nFAILURES:")
        for n, e in FAIL:
            print(f"  {n}: {e}")
        sys.exit(1)
    print("\nAll good. Safe to start a benchmark.")
