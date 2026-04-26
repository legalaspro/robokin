# RoboPlan Examples

Examples using the [RoboPlan](https://github.com/open-planning/roboplan) Optimal Inverse Kinematics (OInK) IK backend with [Viser](https://viser.studio/) for 3D visualization.


## Quick Start

RoboPlan is not available on PyPi, so we recommend using conda.

First, create a new Conda environment.
You can change the name and Python version as you wish.

```bash
conda create -n robokin_env python=3.12 roboplan-python
```

Then, activate the environment.

```bash
conda activate robokin_env
```

Inside the environment (your terminal should now say `(robokin_env)`), you can install this package.

```bash
pip install -e .[viser]
```

You can now run the demo!

```bash
python examples/roboplan/interactive_ik.py 
```
