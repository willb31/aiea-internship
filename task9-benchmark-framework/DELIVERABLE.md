For task 9 I put every algorithm from the last three weeks into one file so they
can all be tested with one command. The framework works. The first thing it told
me is that my task 8 result does not hold up.

The file is rl_benchmark.py. It holds PPO from task 6, both DQNs from task 7,
Double DQN from task 8 and a new Dueling Double DQN, all set up the same way.
Running it with --all trains all five, draws the figures and prints the table.
Adding a sixth means adding one line to a list.

PPO had to change to make the comparison fair. It used a Beta distribution over
the three continuous controls and held each action for 8 frames, and the DQNs
used 5 fixed actions and held them for 4. So the PPO score I have quoted since
task 6 was never measured on the same setup. It now uses the same actions and the
same wrapper. The budget also counts steps taken in the environment rather than
trips round the loop, since PPO wants several environments running and DQN wants
one, and there is a test for that.

| algorithm | best avg100 | final avg100 |
| --- | --- | --- |
| Dueling Double DQN | 744.7 | 713.6 |
| DQN + target net | 672.5 | 672.5 |
| Double DQN | 591.4 | 589.9 |
| DQN 2013 | 546.6 | 546.6 |
| PPO | 502.2 | 221.0 |

Dueling Double DQN won at 744.7. PPO is the odd one. It learns faster than
anything else early and is ahead of the whole field at 62k steps, then falls
apart twice and ends at 221 after peaking at 502. The DQNs are slower but none of
them break.

*[results/figures/01_rewards.png]*

Then the problem. Task 9's dqn2015 and task 8's dqn2015_full are the same
algorithm at the same seed, and I checked that all seventeen settings match. They
reach 672.5 and 609.1 at 200k steps. In task 8 Double DQN was ahead of the target
network there, 645.3 to 609.1, and this week it is behind, 591.4 to 672.5. The
order flipped.

The two runs match exactly for 42 episodes and then split at the first gradient
step. Task 8 pulled 500 held out states from the same random number generator it
used to pick training batches, which shifted every batch it drew after that. Same
algorithm, different stream of random numbers. So last week's 0.4 percent gap was
noise, and that headline should be read as a tie. Two runs of one experiment
differ by 10 percent and trade places.

*[results/figures/03_policies.png]*

The framework can also sweep one setting at a time. On the learning rate, the
2.5e-4 I have used since task 7 came out on top at 107.4, against 84.6 and 38.5.
But 5e-4 was ahead for most of the run and only lost it in the last 8k steps, so
this says which value learns fastest early, not which one ends up best.

*[results/figures/05_sweep_lr.png]*

The roadblock was my own machine again, and this time there is a way out. I have
Nautilus access, namespace aiea-interns, with five A100s shared across it. The
framework writes one cluster job file per algorithm and there is a Dockerfile,
but neither has been tried yet. So these numbers still came off the Mac mini, one
after another, in four and a half hours. Running several seeds on the cluster is
the next step, since one seed is not enough to rank anything.

GitHub repo: https://github.com/willb31/aiea-internship
