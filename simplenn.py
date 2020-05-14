import gym
import tensorflow as tf
import numpy as np

# This is a simple neural netowk to solve the cartpole game.
# This neural netowrk does not focus on learning from past actions

num_inputs = 4
num_hidden = 4
num_outputs = 1  # Probability to go left

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, num_inputs])

hidden_layer_one = tf.layers.dense(
    X, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
hidden_layer_two = tf.layers.dense(
    hidden_layer_one, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)

output_layer = tf.layers.dense(
    hidden_layer_two, num_outputs, activation=tf.nn.sigmoid, kernel_initializer=initializer)

probabilities = tf.concat(axis=1, values=[output_layer, 1-output_layer])
action = tf.multinomial(probabilities, num_samples=1)


init = tf.global_variables_initializer()

step_limit = 500  # The step limit in each episode is 500
epi = 50  # We will run the session for 50 episodes
env = gym.make('CartPole-v1')
avg_steps = []  # List to check how many steps each episode lasts

with tf.Session() as sess:
    init.run()
    for i_episode in range(epi):
        obs = env.reset()
        for step in range(step_limit):
            action_val = action.eval(feed_dict={X: obs.reshape(1, num_inputs)})
            obs, reward, done, info = env.step(
                action_val[0][0])  # Returns 0 or 1

            if(done):
                avg_steps.append(step)
                print("Done after {} steps".format(step))
                break

print('AFTER {} EPISODES, AVERAGE STEPS PER GAME {}'.format(epi, np.mean(avg_steps)))
env.close()
