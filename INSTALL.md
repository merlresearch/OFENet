<!--
Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
## Install Instructions

```bash
$ conda create -n teflon python=3.6 anaconda
$ conda activate teflon
$ conda install cudatoolkit=10.0 cudnn tensorflow-gpu==2.0.0
$ pip install -r requirements.txt
```

### MuJoCo Install Instructions

Install MuJoCo 2.0 from the [official web site](http://www.mujoco.org/index.html).

```bash
$ mkdir ~/.mujoco
$ cd ~/.mujoco
$ wget https://www.roboti.us/download/mujoco200_linux.zip
$ unzip mujoco200_linux.zip
$ mv mujoco200_linux mujoco200
$ cp /path/to/mjkey.txt ./
$ pip install mujoco_py
```

##
