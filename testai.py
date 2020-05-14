import gym

env = gym.make('CartPole-v1')

observation = env.reset()

for _ in range(1000):
    env.render()
    cart_pos, cart_vel, pole_ang, ang_vel = observation

    # We will create a very simple policy where we move the cart to the right if the
    # pole falls to the right and vice versa.
    # This sort of policy will eventually fail.

    if pole_ang > 0:
        action = 1
    else:
        action = 0

    observation, reward, done, info = env.step(action)
