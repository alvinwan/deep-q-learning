# Deep Q-learning Neural Network

This repository provides the original Atari model and a simplified model.

# Install

The project is written in Python 3 and is not guaranteed to successfully backport to Python 2.

(Optional) We recommend setting up a virtual environment.

```
virtualenv dqn --python=python3
source activate dqn/bin/activate
```

Say `$DQN_ROOT` is the root of your repository. Navigate to your root repository.

```
cd $DQN_ROOT
```

We need to setup our Python dependencies.

```
pip install -r requirements.txt
```

Ensure [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) are installed. This project works with CUDA 8.0 and cuDNN 5.1.10.

# Run

```
python run_dqn_atari.py
```

Here are full usage instructions:

```
Usage:
    run_dqn_atari.py [options]

Options:
    --batch-size=<size>     Batch size [default: 32]
    --envid=<envid>         Environment id [default: SpaceInvadersNoFrameskip-v3]
    --model=(atari|simple)  Model to use for training [default: simple]
    --num-filters=<num>     Number of output filters for simple model [default: 64]
    --timesteps=<steps>     Number of timesteps to run [default: 40000000]
```

## Attribution

The code was based on an implementation of Q-learning for Atari
generously provided by Szymon Sidor from OpenAI. The original course
assignment can be found at
http://rll.berkeley.edu/deeprlcourse/docs/hw3.pdf.
