For task 8 I refined last week's DQN with Double Q-learning. The returns came out
about even, the value estimates improved, and the most useful number came from
fixing how I ran task 7 rather than from the new algorithm.

I picked Double DQN because task 7 already measured the problem it solves. The
run without a target network predicted a mean Q of 474.8 while scoring 659.5.
Q-learning builds its training target by taking the best of its own guesses about
the next state, and those guesses are noisy. Taking the best of several noisy
numbers usually lands on one that is too high, so the error builds up instead of
averaging out. Double Q-learning (van Hasselt et al. 2016) splits that into two
jobs. The online network picks the action and the target network says what it is
worth, so an action has to be overestimated by both to survive. In the code it is
one branch. Everything else is held the same and env_and_replay.py is byte
identical to task 7.

I also fixed what I complained about last week. Task 7 capped on wall clock as
well as steps, so the target network run stopped at 329k and I compared it
against a 400k run. This week both stop at exactly 400,000. That mattered more
than the algorithm did. Rerunning last week's winner to its full 400k took it
from 688.5 to 846.7 on unseen tracks, on the same evaluation seeds, so task 7
understated it by 23 percent.

Against that corrected baseline Double DQN is close to a tie. The best 100
episode average went from 782.9 to 786.2. On ten unseen tracks it went from 846.7
to 867.4 and the spread fell from 73.4 to 52.3, so it is a little more consistent
as well as slightly better.

*[results/figures/01_learning_curve.png]*

The value estimates are where the difference shows. I measured these on 500
states frozen after the warm up, the way the Nature paper does it, instead of
reading Q off the training minibatch like last week. Normalized by the return
being achieved at the same moment, the Double DQN estimate is the lower of the
two at 85 percent of the 67 matched checkpoints.

*[results/figures/03_value_ratio.png]*

The roadblock was my own metric. I first wrote that plot as predicted minus
actual and got -330, which I read as huge underestimation until I checked the
units. The value estimate discounts reward that is further in the future and the
episode score does not, so the gap I was plotting was mostly that discounting and
not the algorithm. Comparing the two runs to each other is fine since
both are measured the same way. An absolute number is not, since it would need
the true discounted return from each held out state, and CarRacing cannot be
reset to give me that. It is also one seed, so 85 percent is descriptive.

Speed is the other problem. The Double DQN run took 6.00 hours against 2.25, far
more than one extra forward pass should cost, and throughput fell from 79 to 18
steps per second. That looks like the Mac mini throttling rather than the
algorithm, and it did not affect the comparison, which is the point of capping on
steps. Getting this onto Nautilus is the next step, since throughput has been the
constraint for three weeks now and a GPU would let me run several seeds instead
of one.

GitHub repo: https://github.com/willb31/aiea-internship
