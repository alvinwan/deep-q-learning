"""Play an Atari game using a pretrained model, and save policy.

Usage:
    play.py [options]

Options:
    --n_episodes=<n>    Number of episodes to play [default: 10]
    --envid=<envid>     Environment id [default: SpaceInvadersNoFrameskip-v4]
    --save_path=<path>  Path to saved model
    --logdir=<path>     Path to root of logs directory [default: ./logs]
"""

import docopt
import numpy as np
import tensorflow as tf
import os

from dqn_utils import ReplayBuffer
from dqn_utils import get_wrapper_by_name
from dqn_utils import write_sar_log
from run_dqn_atari import atari_model
from run_dqn_atari import get_env
from run_dqn_atari import get_session


def main():
    arguments = docopt.docopt(__doc__)

    # Run training
    seed = 0  # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(arguments['--envid'], seed)
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
        q_func_vars = tf.get_collection(global_vars, scope='q_func')

        fc1_ph = [var for var in q_func_vars if 'fully_connected/' in var.name][0]
        conv3_ph = [var for var in q_func_vars if 'Conv_2' in var.name][0]
        conv2_ph = [var for var in q_func_vars if 'Conv_1' in var.name][0]
        conv1_ph = [var for var in q_func_vars if 'Conv/' in var.name][0]

        obs_sars, fc1_sars = [], []
        conv3_sars, conv2_sars, conv1_sars = [], [], []

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
                fc1_eval, conv3_eval, conv2_eval, conv1_eval, curr_q_eval = session.run(
                    [fc1_ph, conv3_ph, conv2_ph, conv1_ph, curr_q],
                    {obs_t_ph: r_obs}
                )
                action = np.argmax(curr_q_eval)

                last_obs, reward, done, info = env.step(action)
                episode_reward += reward
                replay_buffer.store_effect(t_obs_idx, action, reward, done)
                obs_sars.append((last_obs, action, reward))
                fc1_sars.append((fc1_eval, action, reward))
                conv3_sars.append((conv3_eval, action, reward))
                conv2_sars.append((conv2_eval, action, reward))
                conv1_sars.append((conv1_eval, action, reward))

                if done:
                    j += 1
                    last_obs = env.reset()
                    episode_rewards = get_wrapper_by_name(env, 'Monitor').get_episode_rewards()
                    if episode_rewards:
                        episode_reward = episode_rewards[-1]
                        if episode_reward < 600:
                            print(' * Reward too low (%d)... resetting.' % episode_reward)
                            obs_sars, fc1_sars = [], []
                            conv3_sars, conv2_sars, conv1_sars = [], [], []
                        else:
                            break

            print(' * Episode %d with reward %d' % (i, episode_reward))
            write_sar_log(obs_sars, logdir, episode_reward, 'obs')
            write_sar_log(fc1_sars, logdir, episode_reward, 'fc1')
            write_sar_log(conv3_sars, logdir, episode_reward, 'conv3')
            write_sar_log(conv2_sars, logdir, episode_reward, 'conv2')
            write_sar_log(conv1_sars, logdir, episode_reward, 'conv1')

            obs_sars, fc1_sars = [], []
            conv3_sars, conv2_sars, conv1_sars = [], [], []


if __name__ == '__main__':
    main()