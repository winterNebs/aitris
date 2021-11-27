from tetris_wrapper import wrapper
import tkinter as tk
from visualizer import vis
import tensorflow as tf
import numpy as np
from numpy.random import default_rng
from tensorflow import keras
from tensorflow.keras import layers, initializers

seed = 1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = epsilon_max - epsilon_min
batch_size = 32
max_steps_per_episode = 10000

env = wrapper()

num_actions = env.action_space


def create_q_network():
    init = initializers.HeUniform()
    inputs = layers.Input(shape=(12,10,1,) )

    layer1 = layers.Conv2D(32, 4, 1, activation="relu", kernel_initializer=init)(inputs)
    layer2 = layers.Conv2D(32, 2, 1, activation="relu", kernel_initializer=init)(layer1)
    #layer2 = layers.Conv1D(16, 1, activation="relu", kernel_initializer=init)(layer1)

    layer3 = layers.Flatten()(layer2)

    layer4 = layers.Dense(1024, activation="relu", kernel_initializer=init)(layer3)
    layer5 = layers.Dense(512, activation="relu", kernel_initializer=init)(layer4)
    action = layers.Dense(num_actions, activation="linear", kernel_initializer=init)(layer5)

    return keras.Model(inputs=inputs, outputs=action)


model = create_q_network()

model_target = create_q_network()

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()

rng = default_rng()

root = tk.Tk()

visualize = True


def toggle():
    global visualize, toggle_btn
    visualize = not visualize
    if toggle_btn.config('relief')[-1] == 'sunken':
        toggle_btn.config(relief="raised")

    else:
        toggle_btn.config(relief="sunken")


toggle_btn = tk.Button(text="togglevisual", relief="raised", command=toggle)
toggle_btn.pack(pady=5)

frame = tk.Frame(root, width=1000, height=1000)
v = vis(root)
best = 0
while True:
    root.title("AI #" + str(1))

    canvas = tk.Canvas(frame, width=400, height=600)
    canvas.pack()
    frame.pack()

    env.reset_tetris(c=canvas)
    state = env.output_formatted_data()
    episode_reward = 0

    for step in range(max_steps_per_episode):
        frame_count += 1
        root.update()
        if visualize:
            v.plotter()
            v.plot_eyes(state, env.tetris.input_log)
            env.tetris.play_field_canvas = canvas
        else:
            env.tetris.play_field_canvas = None
        if frame_count < epsilon_random_frames or epsilon > rng.random():
            # Take random action
            action = rng.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done = env.act_hd(action)
        state_next = np.array(state_next)

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
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))
        if visualize:
            if step % 10 == 0 or done:
                v.add_point(env.tetris.total_score)
                v.set_last(env.tetris.input_log)
                v.add_decay(epsilon)
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


    canvas.destroy()
    best = max(best,episode_reward)
    if episode_count % 100 == 0:
        print(episode_count, best)

def on_close():

    root.destroy()

root.mainloop()