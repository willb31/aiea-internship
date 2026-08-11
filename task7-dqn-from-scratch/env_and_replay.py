"""
The parts of the DQN setup that do not need PyTorch: the discrete action set,
the CarRacing wrapper, the replay memory, and the epsilon schedule.

These live in their own module so test_dqn.py can check them without importing
torch. The replay buffer's index arithmetic is the easiest thing here to get
quietly wrong, so being able to test it in isolation is worth the extra file.
"""

from collections import deque

import numpy as np
import gymnasium as gym


# --------------------------------------------------------------------------
# Action space
# --------------------------------------------------------------------------

# DQN learns Q(s, a) with one output unit per action, so the action space has to
# be finite. CarRacing's is not: steering, gas and brake are all continuous. This
# is the reason last week's PPO used a Beta distribution and this week cannot.
#
# Five actions is the standard minimal set for this environment. Steering is
# full-lock because a discrete agent has no way to express "a little left", and
# gas is 0.8 rather than 1.0 because full throttle in this car mostly produces
# wheelspin.
DISCRETE_ACTIONS = np.array(
    [
        [0.0, 0.0, 0.0],   # 0: coast
        [-1.0, 0.0, 0.0],  # 1: hard left
        [1.0, 0.0, 0.0],   # 2: hard right
        [0.0, 0.8, 0.0],   # 3: accelerate
        [0.0, 0.0, 0.8],   # 4: brake
    ],
    dtype=np.float32,
)


# --------------------------------------------------------------------------
# Environment wrapper
# --------------------------------------------------------------------------
class CarRacingDiscrete:
    """
    CarRacing-v3 with a 5-action discrete interface and the same visual
    pipeline and reward shaping as the task 6 PPO run.

    Keeping the shaping identical matters: the DQN and PPO episode returns are
    only comparable if the environment pays out the same way for both. The one
    thing that differs is action repeat, 4 here against 8 last week, because the
    paper's epsilon schedule is defined per agent decision and 4 keeps this
    closer to the published setup.

    Frames are kept as uint8 rather than the float32 the PPO code used. A float32
    replay buffer of any useful size does not fit in memory; see ReplayBuffer.
    """

    def __init__(self, seed=0, img_stack=4, action_repeat=4, render_mode=None):
        self.env = gym.make("CarRacing-v3", render_mode=render_mode)
        self.env.reset(seed=seed)
        self.img_stack = img_stack
        self.action_repeat = action_repeat
        self.reward_history = deque(maxlen=100)
        self.stack = None

    @staticmethod
    def _to_gray(rgb):
        """RGB uint8 (96, 96, 3) -> uint8 (96, 96). Scaling happens at sample time."""
        gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        return gray.astype(np.uint8)

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        self.reward_history.clear()
        gray = self._to_gray(obs)
        # No history at the start of an episode, so fill the stack with copies.
        self.stack = [gray] * self.img_stack
        return gray, np.array(self.stack)

    def step(self, action_idx):
        action = DISCRETE_ACTIONS[action_idx]
        total_reward = 0.0
        terminated = truncated = False

        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, _ = self.env.step(action)

            # Finishing the track is a success, not a failure. Same bonus as task 6.
            if terminated:
                reward += 100.0

            # A mostly green frame means the car is in the grass.
            if np.mean(obs[:, :, 1]) > 185.0:
                reward -= 0.05

            total_reward += reward
            if terminated or truncated:
                break

        # Cut the episode when the last 100 decisions have all gone badly, so the
        # agent does not spend thousands of steps parked in the grass. Note this
        # is a *time limit* style ending, not a real terminal state, so it is
        # reported separately: bootstrapping through it is correct, and treating
        # it as terminal would teach the agent that being stuck ends the world.
        self.reward_history.append(total_reward)
        stuck = (
            len(self.reward_history) == self.reward_history.maxlen
            and np.mean(self.reward_history) <= -0.1
        )

        gray = self._to_gray(obs)
        self.stack.pop(0)
        self.stack.append(gray)

        return gray, np.array(self.stack), total_reward, terminated, truncated or stuck

    def close(self):
        self.env.close()


# --------------------------------------------------------------------------
# Replay memory
# --------------------------------------------------------------------------
class ReplayBuffer:
    """
    The replay memory D from Section 4, holding the last N transitions.

    Storing whole stacked states would mean 4 frames per transition for the state
    and 4 more for the next state. At 96x96 that is 73 KB per transition, so a
    100k buffer would need 7 GB. Instead this stores each frame exactly once and
    rebuilds the stacks by index at sample time, which costs 9 KB per transition
    and brings the same buffer down to about 0.9 GB.

    The subtlety that makes this worth testing is episode boundaries. Frame t-1
    may belong to the previous episode, in which case stacking it into state t
    would hand the network a state that never existed. _stack walks backwards and
    repeats the oldest valid frame instead of crossing a boundary, which is the
    same thing reset() does at the start of an episode.
    """

    def __init__(self, capacity, img_stack=4, height=96, width=96):
        self.capacity = capacity
        self.img_stack = img_stack
        self.frames = np.zeros((capacity, height, width), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        # terminal = a real end state, bootstrap must be cut
        self.terminals = np.zeros(capacity, dtype=bool)
        # boundary = any episode end, real or a time limit; do not stack across it
        self.boundaries = np.zeros(capacity, dtype=bool)
        self.n_added = 0

    def __len__(self):
        return min(self.n_added, self.capacity)

    @property
    def _oldest(self):
        """Oldest absolute index still present in the ring."""
        return max(0, self.n_added - self.capacity)

    def add(self, frame, action, reward, terminal, boundary):
        i = self.n_added % self.capacity
        self.frames[i] = frame
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminals[i] = terminal
        self.boundaries[i] = boundary
        self.n_added += 1

    def _stack(self, idx):
        """Build the stacked state whose most recent frame is at absolute index idx."""
        out = np.empty((self.img_stack, *self.frames.shape[1:]), dtype=np.uint8)
        out[-1] = self.frames[idx % self.capacity]
        j = idx
        for s in range(self.img_stack - 2, -1, -1):
            prev = j - 1
            # Stop at the start of the ring, or at an episode boundary. Repeating
            # the oldest valid frame mirrors what reset() does.
            if prev < self._oldest or self.boundaries[prev % self.capacity]:
                out[s] = out[s + 1]
            else:
                j = prev
                out[s] = self.frames[j % self.capacity]
        return out

    def sample(self, batch_size, rng):
        """
        Uniform sample, as in the paper. Valid indices need one stored successor
        (for the next state) and are drawn from the part of the ring that has not
        been overwritten.
        """
        lo = self._oldest
        hi = self.n_added - 2  # need idx + 1 to exist
        if hi < lo:
            raise ValueError("replay buffer does not hold a full transition yet")

        idxs = rng.integers(lo, hi + 1, size=batch_size)

        states = np.empty((batch_size, self.img_stack, *self.frames.shape[1:]), dtype=np.uint8)
        next_states = np.empty_like(states)
        for k, idx in enumerate(idxs):
            states[k] = self._stack(idx)
            next_states[k] = self._stack(idx + 1)

        ring = idxs % self.capacity
        return (
            states,
            self.actions[ring],
            self.rewards[ring],
            next_states,
            self.terminals[ring],
        )


# --------------------------------------------------------------------------
# Exploration schedule
# --------------------------------------------------------------------------
def epsilon_at(step, start, end, decay_steps):
    """Linear anneal from `start` to `end` over `decay_steps`, then flat."""
    if step >= decay_steps:
        return end
    return start + (end - start) * (step / decay_steps)
