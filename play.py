"""Play an Atari game using a pretrained model, and save policy.

Usage:
    play.py [options]

Options:
    --n_episodes=<n>    Number of episodes to play [default: 10]
    --envid=<envid>     Environment id [default: SpaceInvaders-v4]
    --save_path=<path>  Path to saved model
    --logdir=<path>     Path to root of logs directory [default: ./logs]
"""

import docopt
import numpy as np
import tensorflow as tf
import os
import os.path
import time

from dqn_utils import ReplayBuffer
from dqn_utils import get_wrapper_by_name
from dqn_utils import write_sar_log
from run_dqn_atari import atari_model
from run_dqn_atari import get_session
from run_dqn_atari import get_custom_env


def main():
    arguments = docopt.docopt(__doc__)

    # Run training
    seed = int(str(time.time())[-5:])
    env = get_custom_env(arguments['--envid'], seed)
    n_episodes = int(arguments['--n_episodes'])
    save_path = arguments['--save_path']
    logdir = arguments['--logdir']

    os.makedirs(logdir, exist_ok=True)

    num_actions = env.action_space.n
    img_h, img_w, img_c = env.observation_space.shape
    frame_history_len = 4
    replay_buffer_size = 1000000
    input_shape = (img_h, img_w, frame_history_len * img_c)
    num_timesteps = 40000000

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    with get_session() as session:

        # set up placeholders
        # placeholder for current observation (or state)
        obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
        # casting to float on GPU ensures lower data transfer times.
        obs_t_float = tf.cast(obs_t_ph, tf.float32) / 255.0

        global_vars = tf.GraphKeys.GLOBAL_VARIABLES
        curr_q = atari_model(obs_t_float, num_actions, scope='q_func')

        obs_sars = []

        saver = tf.train.Saver()
        saver.restore(session, save_path)
        print(' * Restore from', save_path)

        # construct the replay buffer
        replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
        last_obs = env.reset()

        for i in range(n_episodes):
            episode_reward = 0
            j = 0
            while True:
                t_obs_idx = replay_buffer.store_frame(last_obs)

                r_obs = replay_buffer.encode_recent_observation()[
                    np.newaxis, ...]
                curr_q_eval = session.run([curr_q], {obs_t_ph: r_obs})
                action = np.argmax(curr_q_eval)

                last_obs, reward, done, info = env.step(action)
                episode_reward += reward
                replay_buffer.store_effect(t_obs_idx, action, reward, done)
                obs_sars.append(np.hstack((
                    last_obs.reshape((1, -1)),
                    action.reshape((1, 1)),
                    np.array([reward]).reshape((1, 1)))))

                if done:
                    j += 1
                    last_obs = env.reset()
                    episode_rewards = get_wrapper_by_name(env, 'Monitor').get_episode_rewards()
                    if episode_rewards:
                        episode_reward = episode_rewards[-1]
                        if episode_reward < 0:
                            print(' * Reward too low (%d)... resetting.' % episode_reward)
                            obs_sars = []
                        else:
                            break

            print(' * Episode %d with reward %d' % (i, episode_reward))
            write_sar_log(obs_sars, logdir, episode_reward)
            obs_sars = []


if __name__ == '__main__':
    main()