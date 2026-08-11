"""
Task 8: Double DQN refinement of the task 7 agent, on discretized CarRacing-v3.

Task 7 implemented Mnih et al. 2013 from scratch and ran it twice, once as
published and once with the Nature 2015 target network. The target network won,
and the diagnostic that explained why was the value estimate: the run without one
predicted a mean Q of 474.8 while actually scoring 659.5, badly out of step with
what it was achieving.

That is the overestimation bias Q-learning has by construction. The `max` in
`r + gamma * max_a' Q(s', a')` is taken over the same noisy estimates it is
selecting from, so any action whose value happens to be overestimated is exactly
the action the max picks. The error does not average out, it accumulates.

This week refines that with Double Q-learning (van Hasselt, Guez and Silver
2016, arXiv:1509.06461). The fix is to split the two jobs the max is doing:

    standard:  y = r + gamma * Q_target(s', argmax_a' Q_target(s', a'))
    double:    y = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))

The online network picks the action, the target network scores it. Because the
two networks have different errors, an action has to be overestimated by *both*
to survive into the target, which is much rarer.

Everything else is held identical to task 7 on purpose: same seed, same
hyperparameters, same environment wrapper, same architecture, same optimizer.
`env_and_replay.py` is byte-identical to the task 7 file. The only thing that
varies between the two runs this week is the two lines that build `y`.

Two things this fixes about task 7's methodology, both flagged in that write-up:

  1. Runs are capped by step count only. Task 7 capped on wall clock too, so the
     better run also got 18% less data and the comparison was not clean.
  2. Value estimates are measured on a fixed held-out set of states, the way the
     Nature paper's Figure 2 does it, instead of being read off the training
     minibatch. Minibatch Q drifts with whatever the policy is currently
     visiting, so it confounds "the estimate got bigger" with "the agent drove
     somewhere different".

Run:
    python ddqn_carracing.py --steps 400000 --run-name dqn2015_full --target-net
    python ddqn_carracing.py --steps 400000 --run-name ddqn --target-net --double
    python ddqn_carracing.py --eval results/checkpoints/ddqn_best.pt --episodes 10
"""

import argparse
import csv
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env_and_replay import (
    DISCRETE_ACTIONS,
    CarRacingDiscrete,
    ReplayBuffer,
    epsilon_at,
)


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------
class QNetwork(nn.Module):
    """
    The architecture from Section 4.1, with the input resized from the paper's
    84x84 to CarRacing's native 96x96 so no rescaling step is needed.

        conv 16 filters, 8x8, stride 4   ->  23x23
        conv 32 filters, 4x4, stride 2   ->  10x10
        fully connected, 256 units
        fully connected, one output per action

    One output unit per action is the trick from Section 3: a single forward pass
    gives Q for every action at once, so the max over actions costs nothing extra.
    """

    def __init__(self, img_stack=4, n_actions=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(img_stack, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 10 * 10, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        # uint8 [0, 255] -> float [-1, 1], matching the task 6 scaling so the two
        # runs see the same inputs.
        x = x.float() / 128.0 - 1.0
        return self.head(self.features(x))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def bellman_target(online, bootstrap_net, next_states, rewards, terminals, gamma, double):
    """
    The bootstrap target y. This is the one thing that differs between the two
    runs this week, so it lives in its own function where the tests can reach it
    rather than being buried inline in the training loop.

        standard:  y = r + gamma * max_a' Q_boot(s', a')
        double:    y = r + gamma * Q_boot(s', argmax_a' Q_online(s', a'))

    `terminals` is 1.0 only for a true terminal. A stuck-episode timeout is a
    truncation, not a terminal: the episode was cut off by the wrapper, and the
    value of that state is not zero, so its bootstrap must survive.
    """
    with torch.no_grad():
        if double:
            # Online net selects, bootstrap net evaluates. An action now has to
            # be overestimated by both networks to make it into the target.
            next_actions = online(next_states).argmax(dim=1)
            next_q = bootstrap_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
        else:
            # Selection and evaluation both from the bootstrap net, Equation 3.
            next_q = bootstrap_net(next_states).max(dim=1).values

        return rewards + gamma * next_q * (1.0 - terminals)


def open_csv(path, header, resume):
    exists = os.path.exists(path)
    f = open(path, "a" if resume and exists else "w", newline="")
    w = csv.writer(f)
    if not (resume and exists):
        w.writerow(header)
    return f, w


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(args):
    device = pick_device(args.device)
    print(f"device: {device}")
    print(f"target network: {'ON (Nature 2015)' if args.target_net else 'OFF (2013 paper as published)'}")
    print(f"double DQN:     {'ON (van Hasselt 2016)' if args.double else 'OFF (standard max target)'}")
    if args.double and not args.target_net:
        print(
            "WARNING: --double without --target-net does nothing. With no target\n"
            "         network both the selection and the valuation come from the\n"
            "         online net, which is the standard max again. Add --target-net."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env = CarRacingDiscrete(
        seed=args.seed, img_stack=args.img_stack, action_repeat=args.action_repeat
    )
    n_actions = len(DISCRETE_ACTIONS)

    online = QNetwork(args.img_stack, n_actions).to(device)
    target = None
    if args.target_net:
        target = QNetwork(args.img_stack, n_actions).to(device)
        target.load_state_dict(online.state_dict())
        target.eval()

    # The paper uses RMSProp. Adam would probably work better, but the point of
    # this week is to reproduce what is written down.
    optimizer = torch.optim.RMSprop(
        online.parameters(), lr=args.lr, alpha=0.95, eps=0.01
    )

    buffer = ReplayBuffer(args.buffer_size, args.img_stack)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    global_step = 0
    episode = 0
    best_avg = -1e9
    start_time = time.time()

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        online.load_state_dict(ckpt["online"])
        if target is not None and "target" in ckpt:
            target.load_state_dict(ckpt["target"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt["global_step"]
        episode = ckpt["episode"]
        best_avg = ckpt.get("best_avg", -1e9)
        print(f"resumed from {args.resume} at step {global_step}, episode {episode}")

    ep_file, ep_writer = open_csv(
        os.path.join(args.out_dir, f"{args.run_name}_episodes.csv"),
        ["episode", "step", "reward", "length", "running_avg", "epsilon"],
        args.resume,
    )
    up_file, up_writer = open_csv(
        os.path.join(args.out_dir, f"{args.run_name}_updates.csv"),
        ["step", "loss", "mean_q", "max_q", "mean_y", "epsilon", "buffer", "steps_per_sec"],
        args.resume,
    )
    # The held-out value trace, the Nature paper's Figure 2 metric. Kept in its
    # own file because it is sampled on a different cadence to everything else.
    ho_file, ho_writer = open_csv(
        os.path.join(args.out_dir, f"{args.run_name}_holdout.csv"),
        ["step", "holdout_mean_value", "holdout_max_value", "recent_return"],
        args.resume,
    )
    holdout_states = None

    recent = deque(maxlen=100)
    frame, state = env.reset(seed=args.seed)
    ep_reward = 0.0
    ep_len = 0
    loss_val = float("nan")
    mean_q = max_q = float("nan")

    stop_reason = "reached --steps"
    while global_step < args.steps:
        # Wall-clock budget. Throughput is hard to predict ahead of time, so this
        # guarantees the run ends when it is supposed to and leaves a usable
        # checkpoint and log rather than being killed halfway through a write.
        if args.max_hours and (time.time() - start_time) > args.max_hours * 3600:
            stop_reason = f"hit --max-hours ({args.max_hours} h)"
            break

        eps = epsilon_at(global_step, args.eps_start, args.eps_end, args.eps_decay_steps)

        # ---- act, epsilon-greedy (Algorithm 1) ----
        if global_step < args.learn_start or random.random() < eps:
            action = random.randrange(n_actions)
        else:
            with torch.no_grad():
                s = torch.as_tensor(state, device=device).unsqueeze(0)
                action = int(online(s).argmax(dim=1).item())

        next_frame, next_state, reward, terminated, truncated = env.step(action)

        # The frame that was *observed* going into this decision is `frame`, so
        # that is what gets stored alongside the action taken in it.
        buffer.add(
            frame, action, reward,
            terminal=terminated,
            boundary=terminated or truncated,
        )

        ep_reward += reward
        ep_len += 1
        global_step += 1
        frame, state = next_frame, next_state

        if terminated or truncated:
            episode += 1
            recent.append(ep_reward)
            avg = float(np.mean(recent))
            ep_writer.writerow([
                episode, global_step, round(ep_reward, 2), ep_len,
                round(avg, 2), round(eps, 4),
            ])
            ep_file.flush()

            if episode % 10 == 0:
                elapsed = time.time() - start_time
                sps = global_step / max(elapsed, 1e-9)
                print(
                    f"ep {episode:5d} | step {global_step:8d} | reward {ep_reward:8.2f} "
                    f"| avg100 {avg:8.2f} | eps {eps:.3f} | q {mean_q:7.2f} "
                    f"| {sps:6.1f} steps/s | {elapsed/60:.1f} min"
                )

            # Only start tracking "best" once the buffer is warm and the average
            # is over a full window, otherwise the first lucky episode wins.
            if len(recent) == recent.maxlen and avg > best_avg:
                best_avg = avg
                torch.save(
                    {
                        "online": online.state_dict(),
                        "target": target.state_dict() if target else None,
                        "optimizer": optimizer.state_dict(),
                        "global_step": global_step,
                        "episode": episode,
                        "best_avg": best_avg,
                        "args": vars(args),
                    },
                    os.path.join(ckpt_dir, f"{args.run_name}_best.pt"),
                )

            frame, state = env.reset()
            ep_reward = 0.0
            ep_len = 0

        # ---- learn ----
        if global_step >= args.learn_start and global_step % args.train_freq == 0:
            states, actions, rewards, next_states, terminals = buffer.sample(
                args.batch_size, rng
            )

            states_t = torch.as_tensor(states, device=device)
            next_states_t = torch.as_tensor(next_states, device=device)
            actions_t = torch.as_tensor(actions, device=device)
            rewards_t = torch.as_tensor(rewards, device=device)
            terminals_t = torch.as_tensor(terminals, device=device, dtype=torch.float32)

            # Q(s, a) for the action that was actually taken
            q = online(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

            # The bootstrap target, the only thing that differs between the two
            # runs this week. See bellman_target above.
            bootstrap_net = target if target is not None else online
            y = bellman_target(
                online, bootstrap_net, next_states_t, rewards_t, terminals_t,
                args.gamma, args.double,
            )

            # Huber rather than squared error. The paper's own follow-up notes
            # squared error blows up when rewards are unclipped, which they are
            # here by design.
            loss = F.smooth_l1_loss(q, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), args.max_grad_norm)
            optimizer.step()

            loss_val = float(loss.item())
            with torch.no_grad():
                mean_q = float(q.mean().item())
                max_q = float(q.max().item())
                mean_y = float(y.mean().item())

            if target is not None and global_step % args.target_sync == 0:
                target.load_state_dict(online.state_dict())

            if global_step % args.log_every == 0:
                elapsed = time.time() - start_time
                up_writer.writerow([
                    global_step, round(loss_val, 5), round(mean_q, 4), round(max_q, 4),
                    round(mean_y, 4),
                    round(eps, 4), len(buffer),
                    round(global_step / max(elapsed, 1e-9), 2),
                ])
                up_file.flush()

            # ---- held-out value estimate ----
            # Frozen once, right after the warm-up, from the random-policy states
            # sitting in the buffer. The set never changes again, so the trace
            # measures the network's opinion of a fixed set of positions rather
            # than of wherever the current policy happens to be driving.
            if holdout_states is None:
                sampled, _, _, _, _ = buffer.sample(args.holdout_size, rng)
                holdout_states = torch.as_tensor(sampled, device=device)
                print(f"froze {args.holdout_size} held-out states at step {global_step}")

            if global_step % args.holdout_every == 0:
                online.eval()
                with torch.no_grad():
                    hq = online(holdout_states).max(dim=1).values
                    ho_mean = float(hq.mean().item())
                    ho_max = float(hq.max().item())
                online.train()
                ho_writer.writerow([
                    global_step, round(ho_mean, 4), round(ho_max, 4),
                    round(float(np.mean(recent)), 2) if recent else "",
                ])
                ho_file.flush()

        if global_step % args.save_every == 0:
            torch.save(
                {
                    "online": online.state_dict(),
                    "target": target.state_dict() if target else None,
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "episode": episode,
                    "best_avg": best_avg,
                    "args": vars(args),
                },
                os.path.join(ckpt_dir, f"{args.run_name}_latest.pt"),
            )

    torch.save(
        {
            "online": online.state_dict(),
            "target": target.state_dict() if target else None,
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "episode": episode,
            "best_avg": best_avg,
            "args": vars(args),
        },
        os.path.join(ckpt_dir, f"{args.run_name}_final.pt"),
    )

    ep_file.close()
    up_file.close()
    ho_file.close()
    env.close()

    total = time.time() - start_time
    print(
        f"\ndone ({stop_reason}). {global_step} steps over {episode} episodes in "
        f"{total/3600:.2f} h ({total/60:.1f} min). best avg100 {best_avg:.2f}"
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(args):
    """Greedy rollouts from a checkpoint, no exploration, on unseen tracks."""
    device = pick_device(args.device)
    ckpt = torch.load(args.eval, map_location=device)
    online = QNetwork(args.img_stack, len(DISCRETE_ACTIONS)).to(device)
    online.load_state_dict(ckpt["online"])
    online.eval()

    env = CarRacingDiscrete(
        seed=args.seed + 10_000,
        img_stack=args.img_stack,
        action_repeat=args.action_repeat,
    )

    scores = []
    for i in range(args.episodes):
        _, state = env.reset(seed=args.seed + 10_000 + i)
        total = 0.0
        done = False
        while not done:
            with torch.no_grad():
                s = torch.as_tensor(state, device=device).unsqueeze(0)
                action = int(online(s).argmax(dim=1).item())
            _, state, reward, terminated, truncated = env.step(action)
            total += reward
            done = terminated or truncated
        scores.append(total)
        print(f"  eval episode {i+1}: {total:.2f}")

    env.close()
    print(f"\nmean {np.mean(scores):.2f}  std {np.std(scores):.2f}  over {len(scores)} episodes")
    return scores


# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="DQN from scratch on discretized CarRacing-v3")
    p.add_argument("--steps", type=int, default=400_000,
                   help="agent decisions, not environment frames")
    p.add_argument("--max-hours", type=float, default=None,
                   help="wall-clock budget; stops cleanly and saves whatever it has")
    p.add_argument("--run-name", type=str, default="ddqn")
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--img-stack", type=int, default=4, help="paper used 4")
    p.add_argument("--action-repeat", type=int, default=4, help="paper used 4")

    p.add_argument("--buffer-size", type=int, default=100_000,
                   help="paper used 1e6 frames; 1e5 here to fit in memory")
    p.add_argument("--batch-size", type=int, default=32, help="paper used 32")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--train-freq", type=int, default=4,
                   help="gradient steps happen every N agent decisions")
    p.add_argument("--learn-start", type=int, default=5_000,
                   help="pure random acting until the buffer has this many transitions")
    p.add_argument("--max-grad-norm", type=float, default=10.0)

    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.1, help="paper annealed to 0.1")
    p.add_argument("--eps-decay-steps", type=int, default=100_000)

    p.add_argument("--target-net", action="store_true",
                   help="add the Nature 2015 target network (off = 2013 paper as published)")
    p.add_argument("--target-sync", type=int, default=1_000,
                   help="copy online weights into the target every N steps")

    p.add_argument("--double", action="store_true",
                   help="Double DQN target: online net selects the action, "
                        "bootstrap net values it (van Hasselt et al. 2016)")

    p.add_argument("--holdout-size", type=int, default=500,
                   help="states frozen after warm-up for the value trace")
    p.add_argument("--holdout-every", type=int, default=5_000,
                   help="how often to score the held-out set")

    p.add_argument("--log-every", type=int, default=1_000)
    p.add_argument("--save-every", type=int, default=25_000)
    p.add_argument("--resume", type=str, default=None)

    p.add_argument("--eval", type=str, default=None, help="path to a checkpoint")
    p.add_argument("--episodes", type=int, default=10)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
