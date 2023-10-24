<!--
Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# OFENet

## Features

OFENet is a feature extractor network for low-dimensional data to improve performance of Reinforcement Learning.
It can be combined with algorithms such as PPO, DDPG, TD3, and SAC.

This repository contains OFENet implementation, RL algorithms, and hyperparameters, which
we used in our paper. We ran these codes on Ubuntu 18.04 & GeForce 1060.

## Installation

See [INSTALL.md](INSTALL.md).

## Usage

See [GETTING_STARTED.md](GETTING_STARTED.md)).

## Testing

Explain how others can test your code (eg run tests, demos, etc)

## Citation

If you use the software, please cite the following  ([TR2020-083](https://merl.com/publications/TR2020-083)):

```bibTeX
@inproceedings{ota2020can,
  title={Can increasing input dimensionality improve deep reinforcement learning?},
  author={Ota, Kei and Oiki, Tomoaki and Jha, Devesh and Mariyama, Toshisada and Nikovski, Daniel},
  booktitle={International conference on machine learning},
  pages={7424--7433},
  year={2020},
  organization={PMLR}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:

```
Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: AGPL-3.0-or-later
```

`tfrl` was adapted from https://github.com/deepmind/trfl (`Apache-2.0` license as found
in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)).

`teflon/policy/PPO.py` was adapted from https://github.com/keiohta/tf2rl (`MIT` license as found
in [LICENSES/MIT.txt](LICENSES/MIT.txt)).

`util/gin_tf_external.py` and `util/gin_utils.py` were adapted from https://github.com/google/gin-config (`Apache-2.0` license as found in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)).

## Contact

If you have problem running codes, please contact Kei Ota (ota.kei@ds.mitsubishielectric.co.jp) or Devesh K. Jha (jha@merl.com)
