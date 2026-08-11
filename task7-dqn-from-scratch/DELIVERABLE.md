For task 7 I built DQN from scratch and trained it on CarRacing-v3. The paper the
task links is the 2013 version, which has no target network, so I ran it twice,
once as written and once with the target network the 2015 paper added. Both
learned to drive, and the gap between them is the interesting part.

I first read the DQN paper (Mnih et al. 2013) and made a slide deck summing it
up. Then I wrote dqn_carracing.py straight from the paper, without stable
baselines 3 or any other RL library. Experience replay, the epsilon greedy
schedule from 1.0 down to 0.1, the Q learning target and the conv net from
section 4.1 are all written out in PyTorch.

The environment had to change from last week. DQN puts one output per action and
takes the max over them, so the action space has to be finite, and CarRacing's
steering, gas and brake are all continuous. That is the same reason last week's
PPO used a Beta distribution. I cut it down to five actions: coast, hard left,
hard right, accelerate and brake. Everything else about the environment is the
same as task 6, so the scores compare directly. I also skipped the reward
clipping the paper uses, since CarRacing pays -0.1 per frame and +1000/N per
track tile and clipping would make those the same size.

The first run was the paper as published. It did 399,847 steps over 1,886
episodes in 1.95 hours and reached a 100 episode average of 690.5. The second
added a target network synced every 1,000 steps and reached 779.6 on 328,660
steps, because it hit my 2.5 hour cap first, so it did better on 18 percent less
data. On 5 new tracks the first averaged 525.9 and the second 688.5. Both are far
past the 111.2 PPO managed last week.

*[results/figures/01_learning_curve.png]*

The predicted Q values show why the target network helps. Without one the network
computes its own regression target, so any overestimate feeds back into the next
target. Mean predicted Q climbed to 474.8 while that run was scoring 659.5, and
with the target network it only reached 357.5 while scoring 763.5. The paper
version predicted about a third more value while driving worse, which is the
overestimation the 2015 paper was written to fix.

*[results/figures/02_q_values.png]*

The roadblocks were mostly speed. Throughput fell over both runs, from 64 steps
per second to 57 on the first and 53 to 41 on the second, which is why the second
ran out of time. The Mac mini is probably throttling. That makes the comparison
less clean than I wanted, since the better run also got less data, though that
cuts against the target network rather than for it. Neither curve had flattened
when they stopped, so next time I would cap both by step count only and run them
longer.

GitHub repo: https://github.com/willb31/aiea-internship
