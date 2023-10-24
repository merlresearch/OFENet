<!--
Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
## Examples

To train an agent with OFENet, run the below commands at the project root.

```bash
$ export PYTHONPATH=.
$ python3 teflon/tool/eager_main.py --policy SAC \
                                  --env HalfCheetah-v2 \
                                  --gin ./gins/HalfCheetah.gin \
                                  --seed 0
```

If you want to combine OFENet with TD3 or DDPG, change the policy like

```bash
$ export PYTHONPATH=.
$ python3 teflon/tool/eager_main.py --policy TD3 \
                                  --env HalfCheetah-v2 \
                                  --gin ./gins/HalfCheetah.gin \
                                  --seed 0
```

When you want to run an agent in another environment, change the policy and
the hyperparameter file (.gin).

```bash
$ python3 teflon/tool/eager_main.py --policy SAC \
                                  --env Walker2d-v2  \
                                  --gin ./gins/Walker2d.gin \
                                  --seed 0
```

When you don't specify a gin file, you train an agent with raw observations.

```bash
$ python3 teflon/tool/eager_main.py --policy SAC \
                                  --env HalfCheetah-v2 \
                                  --seed 0
```

ML-SAC is trained with the below command.

```bash
$ python3 teflon/tool/eager_main.py --policy SAC \
                                  --env HalfCheetah-v2 \
                                  --gin ./gins/Munk.gin \
                                  --seed 0
```

## Retrieve results

`eager_main.py` generates a log file under "log" directory.
You can watch the result of an experiment with tensorboard.

```bash
$ tensorboard --logdir ./log
```
