
.PHONY: sample

sample:
	python run_dqn_atari.py --learning-starts=40000 --timesteps=80000 --model=fesimple --restore=./models/init.ckpt
