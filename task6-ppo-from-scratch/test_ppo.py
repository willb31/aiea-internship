"""
Checks that do not need a GPU or a trained agent.

    python test_ppo.py

Covers the two things easiest to get silently wrong: the GAE recursion and the
conv stack's output size. Run it before starting a long training job.
"""

import numpy as np


# ---------------------------------------------------------------- GAE
def reference_gae(rewards, values, dones, last_value, gamma, lam):
    """Straightforward loop, written independently of the buffer implementation."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float64)
    running = 0.0
    for t in reversed(range(n)):
        next_v = last_value if t == n - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * mask - values[t]
        running = delta + gamma * lam * mask * running
        adv[t] = running
    return adv


def test_gae_matches_reference():
    from ppo_carracing import RolloutBuffer

    rng = np.random.default_rng(0)
    n, gamma, lam = 64, 0.99, 0.95

    buf = RolloutBuffer(n, 1, img_stack=1, device="cpu")
    rewards = rng.normal(size=n).astype(np.float32)
    values = rng.normal(size=n).astype(np.float32)
    dones = (rng.random(n) < 0.1).astype(np.float32)
    buf.rewards[:, 0] = rewards
    buf.values[:, 0] = values
    buf.dones[:, 0] = dones
    buf.ptr = n

    last_value = 0.37
    adv, returns = buf.compute_gae([last_value], gamma, lam)
    adv, returns = adv[:, 0], returns[:, 0]
    expected = reference_gae(rewards, values, dones, last_value, gamma, lam)

    assert np.allclose(adv, expected, atol=1e-4), "GAE does not match the reference loop"
    assert np.allclose(returns, adv + values, atol=1e-4), "returns should be advantages + values"
    print("ok  GAE matches an independent implementation")


def test_gae_terminal_cuts_the_bootstrap():
    """After a terminal step, nothing beyond it should leak backwards."""
    from ppo_carracing import RolloutBuffer

    n = 5
    buf = RolloutBuffer(n, 1, img_stack=1, device="cpu")
    buf.rewards[:, 0] = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    buf.values[:, 0] = np.zeros(n, dtype=np.float32)
    buf.dones[:, 0] = np.array([0, 0, 1, 0, 0], dtype=np.float32)
    buf.ptr = n

    adv, _ = buf.compute_gae([0.0], gamma=0.99, lam=0.95)
    adv = adv[:, 0]

    # step 2 is terminal, so its advantage is just its own reward
    assert abs(adv[2] - 1.0) < 1e-5, f"terminal step should not bootstrap, got {adv[2]}"
    # step 1 sees step 2 but the chain stops there
    assert adv[1] > adv[2], "earlier steps should accumulate more return"
    print("ok  terminal states cut the bootstrap")


def test_gae_lambda_one_is_discounted_return():
    """lam = 1 with zero baseline should give the plain discounted return."""
    from ppo_carracing import RolloutBuffer

    n, gamma = 6, 0.9
    buf = RolloutBuffer(n, 1, img_stack=1, device="cpu")
    buf.rewards[:, 0] = np.ones(n, dtype=np.float32)
    buf.values[:, 0] = np.zeros(n, dtype=np.float32)
    buf.dones[:, 0] = np.zeros(n, dtype=np.float32)
    buf.ptr = n

    adv, _ = buf.compute_gae([0.0], gamma=gamma, lam=1.0)
    adv = adv[:, 0]
    expected_first = sum(gamma ** k for k in range(n))
    assert abs(adv[0] - expected_first) < 1e-4, f"{adv[0]} != {expected_first}"
    print("ok  lambda = 1 reduces to the discounted return")


def test_gae_is_independent_per_actor():
    """
    The whole risk of vectorizing is one actor's returns bleeding into another's.
    Run 4 actors together, then run each one alone, and require identical results.
    """
    from ppo_carracing import RolloutBuffer

    rng = np.random.default_rng(7)
    T, N, gamma, lam = 32, 4, 0.99, 0.95

    rewards = rng.normal(size=(T, N)).astype(np.float32)
    values = rng.normal(size=(T, N)).astype(np.float32)
    dones = (rng.random((T, N)) < 0.15).astype(np.float32)
    last_values = rng.normal(size=N).astype(np.float32)

    vec = RolloutBuffer(T, N, img_stack=1, device="cpu")
    vec.rewards[:] = rewards
    vec.values[:] = values
    vec.dones[:] = dones
    vec.ptr = T
    adv_vec, ret_vec = vec.compute_gae(last_values, gamma, lam)

    for i in range(N):
        solo = RolloutBuffer(T, 1, img_stack=1, device="cpu")
        solo.rewards[:, 0] = rewards[:, i]
        solo.values[:, 0] = values[:, i]
        solo.dones[:, 0] = dones[:, i]
        solo.ptr = T
        adv_solo, ret_solo = solo.compute_gae([last_values[i]], gamma, lam)
        assert np.allclose(adv_vec[:, i], adv_solo[:, 0], atol=1e-5), \
            f"actor {i} advantages differ when batched"
        assert np.allclose(ret_vec[:, i], ret_solo[:, 0], atol=1e-5), \
            f"actor {i} returns differ when batched"
    print("ok  each actor's GAE is unaffected by the others")


def test_flatten_preserves_pairing():
    """Flattening (T, N) to T*N must keep each observation with its own action."""
    from ppo_carracing import RolloutBuffer

    T, N = 5, 3
    buf = RolloutBuffer(T, N, img_stack=1, device="cpu")
    for t in range(T):
        for i in range(N):
            tag = t * 10 + i
            buf.obs[t, i] = tag          # every pixel carries the tag
            buf.actions[t, i] = [tag, tag, tag]
    buf.ptr = T

    adv = np.zeros((T, N), dtype=np.float32)
    ret = np.zeros((T, N), dtype=np.float32)
    obs, actions, _, _, _ = buf.to_tensors(adv, ret)

    obs_tags = obs.reshape(T * N, -1)[:, 0].numpy()
    action_tags = actions[:, 0].numpy()
    assert np.array_equal(obs_tags, action_tags), \
        "flattening scrambled the observation/action pairing"
    print("ok  flattening keeps every observation with its own action")


# ---------------------------------------------------------------- shapes
def test_conv_output_is_256():
    """
    Work the conv arithmetic by hand so a bad kernel size is caught here rather
    than by a shape error twenty minutes into training.
    """
    size = 96
    layers = [(4, 2), (3, 2), (3, 2), (3, 2), (3, 1), (3, 1)]  # (kernel, stride)
    for k, s in layers:
        size = (size - k) // s + 1
    assert size == 1, f"conv stack ends at {size}x{size}, expected 1x1"
    print("ok  conv stack reduces 96x96 to 1x1 (256 features)")


def test_action_mapping_stays_in_range():
    """Beta samples live in [0,1]; check the mapping lands inside the env's box."""
    a = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.3, 0.7]])
    mapped = np.stack([a[:, 0] * 2 - 1, a[:, 1], a[:, 2]], axis=1)
    assert mapped[:, 0].min() >= -1 and mapped[:, 0].max() <= 1, "steering out of range"
    assert mapped[:, 1:].min() >= 0 and mapped[:, 1:].max() <= 1, "gas/brake out of range"
    assert np.allclose(mapped[2], [0.0, 0.3, 0.7])
    print("ok  action mapping stays inside the CarRacing action space")


def test_torch_forward_pass():
    """Only runs if torch is installed. Confirms real tensor shapes end to end."""
    try:
        import torch
    except ImportError:
        print("--  torch not installed, skipping the forward-pass check")
        return

    from ppo_carracing import ActorCritic

    net = ActorCritic(img_stack=4)
    x = torch.randn(3, 4, 96, 96)
    alpha, beta, value = net(x)
    assert alpha.shape == (3, 3), alpha.shape
    assert beta.shape == (3, 3), beta.shape
    assert value.shape == (3, 1), value.shape
    assert (alpha > 1).all() and (beta > 1).all(), "Beta parameters must exceed 1"

    action, log_prob, v = net.act(x)
    assert action.shape == (3, 3) and log_prob.shape == (3,) and v.shape == (3,)
    assert (action >= 0).all() and (action <= 1).all(), "Beta samples must lie in [0,1]"

    lp, ent, val = net.evaluate(x, action)
    assert lp.shape == (3,) and ent.shape == (3,) and val.shape == (3,)
    assert torch.isfinite(lp).all() and torch.isfinite(ent).all()
    print("ok  forward pass, sampling and evaluation all return finite, correctly shaped tensors")


if __name__ == "__main__":
    test_gae_matches_reference()
    test_gae_terminal_cuts_the_bootstrap()
    test_gae_lambda_one_is_discounted_return()
    test_gae_is_independent_per_actor()
    test_flatten_preserves_pairing()
    test_conv_output_is_256()
    test_action_mapping_stays_in_range()
    test_torch_forward_pass()
    print("\nall checks passed")
