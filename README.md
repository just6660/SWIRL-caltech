# SWIRL

Code for "Inverse Reinforcement Learning with Switching Rewards and History Dependency for Characterizing Animal Behaviors".

## Simulated Gridworld 

For simulated 5x5 gridworld experiment, run:

```
cd gw5/swirl/
python run_gw5.py
```

## Labyrinth

For labyrinth experiment (Rosenberg et al., 2021), run:
```
cd labyrinth/swirl/
python run_labyrinth.py
```

## Spontaneous Behavior

For mouse spontaneous behavior experiment (Markowitz et al., 2023), run:
```
cd spontda/swirl/
python run_da.py
```

## smm-based ARHMM code

In labyrinth and mouse spontaneous behavior experiments, we initialize SWIRL parameters using ARHMM, implemented based on the [`ssm`](https://github.com/lindermanlab/ssm) library. Our modified ARHMM implementation (`hmm_old.py`, `observations.py`, `messages.py`,  `primitives.py`, `transitions.py`,  `stats.py`, `swirl.py`) is provided in `ssm/`. To use it, add these scripts to the `ssm/ssm/` folder of the original [`ssm`](https://github.com/lindermanlab/ssm) repository before installation. 
