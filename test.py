import gym
env = gym.make('CartPole-v1')

# Gym observations

# The environment.step() function returns useful objects for our learning agent
# There are 4 KEY THINGS returned by the step() function
# One of them is the OBSERVATION variable.
# Observation is the environment specific information representing environment observations.
# For eg- Angle, velocity, game state etc
# In the case of the cart pole, we see that it is a 4d array with info regarding
# Angular velocity, velocity, angle of the pole and the displacement

# The next variable returned is the REWARD
# It is the amount of reward achieved by previous action
# Scale varies depending upon the environment but agen should want to increase reward level

# Third variable returned by step() is the DONE variable
# This is a boolean indicating whether the env needs to be reset
# Eg- Game lost, pole tipped over, game won etc

# Lastly, an INFO object is returned
# It is a dictionary object with diagnostic information ,for debugging etc



# Set environment to initial value

# Print initial observation
observation = env.reset()
print(observation)

# Render environment over several timesteps

for _ in range(2):
    env.render()

    # Take some random action from the available action space available
    # Here- Go left or right (ie,0 or 1)
    # Provide that random step into the environment

    action = env.action_space.sample()

    # Tuple unpacking to grab the returned objects
    observation, reward, done, info = env.step(action)

    print('Performed one random action')
    print('\n')
    print(observation)
    print('\n')
    print(reward)
    print('\n')
    print(done)
    print('\n')
    print(info)
    print('\n')
