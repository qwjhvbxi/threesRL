# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:33:20 2021

@author: Riccardo Iacobucci
"""

import threes as game
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pickle
import os

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# input: binary variables. 4*4*15 = 240 (board) + 10 (next piece: 1,2,3,6,6-12,12-96,...)

# foldername = 'tmp10'; # simplified. reward based on adjusted difference in score and -1 for illegal moves
foldername = 'tmp13'; # reward based on adjusted difference in score and -1 for illegal moves

simplified = 0;

## option 1: flat input
num_inputs = 250
num_actions = 4
num_hidden1 = 250
num_hidden2 = 250

# NN structure
inputs = layers.Input(shape=(num_inputs,))
common1 = layers.Dense(num_hidden1, activation="relu")(inputs)
common2 = layers.Dense(num_hidden2, activation="relu")(common1)
action = layers.Dense(num_actions, activation="softmax")(common2)
critic = layers.Dense(1)(common2)

checkpoint_filepath_status = foldername+'/checkpoint_status'
checkpoint_filepath = foldername+'/checkpoint_weights'

gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 2000;
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

#env = game.tre(); # normal
env = game.tre(simplified); # simplified

model = keras.Model(inputs=inputs, outputs=[action, critic])

optimizer = keras.optimizers.Adam(learning_rate=0.00025,clipnorm=1)
huber_loss = keras.losses.Huber()

action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0
reward_progress = [];
maxnumber_progress = [];

# load existing model
if os.path.exists(checkpoint_filepath_status):
    with open(checkpoint_filepath_status,'rb') as f:
        action_probs_history,critic_value_history,rewards_history,running_reward,episode_count,reward_progress,maxnumber_progress = pickle.load(f)
    model.load_weights(checkpoint_filepath)

else:
    f = open(checkpoint_filepath_status,'w+');
    f.close();

maxnum = 0;
numfinished = 0;

while True:
    state = env.reset();
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])
            
            probs = np.squeeze(action_probs);
            
            # sample action from action probability distribution
            action = np.random.choice(num_actions, p=probs)
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # apply the sampled action
            state, reward, done = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward
            
            if timestep==max_steps_per_episode-1:
                numfinished = numfinished + 1;
                # print('finished')
            
            if done:
                break
            
        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        
        maxnum = np.max((maxnum,env.maxnum())).astype(int);

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()
        
        reward_progress.append(running_reward);

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}. Max number: {}. Num. finished: {}"
        print(template.format(running_reward, episode_count, maxnum, numfinished))
        maxnumber_progress.append(maxnum);
        maxnum = 0;
        numfinished = 0;
        
    if episode_count % 100 == 0:
        
        # save model
        # model.save_weights(checkpoint_path.format(epoch=0))
        model.save_weights(checkpoint_filepath.format(epoch=0))
        f = open(checkpoint_filepath_status,'wb');
        pickle.dump([action_probs_history,critic_value_history,rewards_history,
        running_reward,episode_count,reward_progress,maxnumber_progress],f);
        f.close();
        print('checkpoint saved.')
        
        # print progress
        avgmax = running_mean(maxnumber_progress,50);
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(running_mean(reward_progress,100), 'k-')
        ax1.grid(color='k', axis='y',linestyle='-', linewidth=0.1)
        ax2.plot(np.linspace(0,len(reward_progress),num=len(avgmax)),avgmax, 'r-')
        
        # plt.plot(running_mean(reward_progress,100))
        # plt.plot(np.linspace(0,len(reward_progress),num=len(avgmax)),avgmax)
        plt.show()  
        
        # plt.plot(reward_progress)
        # plt.show();

    # Use with profiler
    # if episode_count == 50:
    #     break

    if running_reward > 100000:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break

plt.plot(reward_progress)
plt.plot(running_mean(reward_progress,500))
plt.show()

print(np.max(reward_progress))
print(np.argmax(reward_progress))

plt.plot(maxnumber_progress)
plt.plot(running_mean(maxnumber_progress,100))
plt.show()



# play a sample game
state = env.reset()
action_history = [];
totalreward = 0;
for timestep in range(1, max_steps_per_episode):
    
    #predictions = model.predict(state)
    
    env.show();
    state = tf.convert_to_tensor(state)
    state = tf.expand_dims(state, 0)
    
    action_probs, critic_value = model(state)
    
    value = (np.squeeze(critic_value)*100).astype(int)
    print("Next: {} - Action probs: {} - Critic value: {} ".format(env.nextpiece,np.round(np.squeeze(action_probs),2),value/100))
    
    activelines,_ = env.possiblemoves();
    possiblemoves = np.sum(activelines,1)>0;    
    newmoves = (np.squeeze(action_probs)+0.0001)*possiblemoves;
    
    # Sample action from action probability distribution
    action = np.argmax(newmoves)
    
    action_history.append(action);

    # Apply the sampled action in our environment
    state, reward, done = env.step(action)
    
    print("Chosen move: {} - Reward: {}".format(action,reward))
    
    totalreward = totalreward+reward;
    
    
    if done:
        break
    
print(totalreward)

