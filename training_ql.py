# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 15:43:54 2021

@author: Riccardo Iacobucci
"""

# from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import threes as game
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os

# Configuration paramaters for the whole setup
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 100

simplified = 0; # play simplified game?

env = game.tre(simplified) 

num_inputs = 250 # binary variables. 4*4*15 = 240 (board) + 10 (next piece: 1,2,3,6,6-12,12-96,...)
num_hidden1 = 125
# num_hidden2 = 125
num_hidden2 = 60
num_actions = 4


def create_q_model():
    inputs = layers.Input(shape=(num_inputs,))
    common1 = layers.Dense(num_hidden1, activation="relu")(inputs)
    common2 = layers.Dense(num_hidden2, activation="relu")(common1)
    action = layers.Dense(num_actions, activation="softmax")(common2)

    return keras.Model(inputs=inputs, outputs=action)


# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)


checkpoint_filepath_status = 'tmp/checkpoint_status'
checkpoint_filepath = 'tmp/checkpoint_weights'
# checkpoint_filename = 'tmp/checkpoint.sav'
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=False,
#     monitor='val_accuracy',
#     mode='max',
#     save_best_only=False)


# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
reward_progress = [];


# load existing model
if os.path.exists(checkpoint_filepath_status):
    f = open(checkpoint_filepath_status,'rb');
    action_history,state_history,state_next_history,rewards_history,done_history,episode_reward_history,running_reward,episode_count,frame_count,reward_progress = pickle.load(f)
    f.close();    
    model.load_weights(checkpoint_filepath)

else:
    f = open(checkpoint_filepath_status,'w+');
    f.close();


# Number of frames to take random action and observe output
epsilon_random_frames = 500
# Number of frames for exploration
epsilon_greedy_frames = 1000
# Maximum replay length
max_memory_length = 1000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 2000  # 10000   
# Using huber loss for stability
loss_function = keras.losses.Huber()
maxnum = 0;


while True:  # Run until solved
    state = np.array(env.reset())
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.
        frame_count += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done = env.step(action)
        state_next = np.array(state_next)
        
        # env.show()
        # print(reward)
        
        maxnum = np.max((maxnum,env.maxnum())).astype(int);

        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}, max number {}"
            print(template.format(running_reward, episode_count, frame_count, maxnum))
            maxnum = 0;
            
            # save model
            # model.save_weights(checkpoint_path.format(epoch=0))
            model.save_weights(checkpoint_filepath.format(epoch=0))
            f = open(checkpoint_filepath_status,'wb');
            pickle.dump([action_history,state_history,state_next_history,
            rewards_history,done_history,episode_reward_history,
            running_reward,episode_count,frame_count,reward_progress],f);
            f.close();

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break


    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    if running_reward > 99:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break
    
    reward_progress.append(running_reward);
    



plt.plot(reward_progress)


# plot
state = env.reset()
act_history = [];
for timestep in range(1, max_steps_per_episode):
    
    #predictions = model.predict(state)
    
    env.show();
    state = tf.convert_to_tensor(state)
    state = tf.expand_dims(state, 0)

    action_probs = model(state, training=False)
    
    print(np.round(np.squeeze(action_probs),2))
    print(env.nextpiece)

    # Sample action from action probability distribution
    action = np.argmax(np.squeeze(action_probs))
    
    act_history.append(action);

    # Apply the sampled action in our environment
    state, reward, done = env.step(action)
    
    if done:
        break
    
