import random
from tetris_wrapper import wrapper
import tkinter as tk
from visualizer import vis
import tensorflow as tf
import numpy as np
from numpy.random import default_rng
from tensorflow import keras
from tensorflow.keras import layers

rng = default_rng()
seed = 42
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = epsilon_max - epsilon_min
batch_size = 32
max_steps_per_episode = 1000


def create_q_network():
    inputs = layers.Input(shape=(10, 29, 1,))
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, fctivation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(wrapper.action_space, activation="linear")(layer5)

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
# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()

while True:
    wr = wrapper()
    wr.reset_tetris()
    state = np.array(wr.output_data())
    episode_reward = 0

    for step in range(max_steps_per_episode):
        frame_count += 1

        if frame_count < epsilon_random_frames or epsilon > rng.random():
            action = rng.ran(low=0, high=wrapper.action_space, size=1)[0]


def run_ai(parent_window, frame):
    v = vis(root)
    seed = 0
    wr = wrapper()
    state_size = 290
    action_size = wr.action_space

    agent = Network(state_size, action_size)
    # train(agent, [6, 9, 0, 5, 8, 0, 9, 2, 0, 9, 0, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 0, 8, 0, 5, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 0, 6, 0, 2, 2, 0, 5, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 0, 2, 2, 2, 0, 0, 6, 2, 0, 9, 0, 1, 0, 9, 6, 1, 0, 6, 8, 0, 5, 9, 2, 0, 8, 1, 0, 5, 0, 6, 2, 0, 9, 0, 8, 0, 8, 1, 0, 9, 6, 0, 5, 0, 8, 0, 6, 9, 0, 2, 2, 0, 8, 1, 0, 2, 2, 0, 6, 0, 7, 1, 0, 1, 0, 1, 0, 7, 0, 7, 8, 1, 0, 6, 8, 0, 5, 1, 0, 9, 0, 2, 0, 6, 9, 0, 6, 9, 2, 0, 8, 1, 0, 5, 2, 0, 2, 6, 2, 0, 6, 8, 0, 1, 0, 0, 9, 2, 0, 0, 2, 2, 6, 0, 5, 1, 0, 6, 8, 0, 6, 9, 0, 6, 8, 1, 0, 5, 9, 0, 2, 2, 5, 2, 0, 2, 0, 5, 1, 0, 0, 9, 2, 0, 6, 9, 0, 7, 9, 2, 0, 5, 8, 1, 0, 6, 8, 0, 6, 9, 0, 8, 1, 1, 0, 0, 0, 5, 1, 0, 7, 0, 6, 8, 0, 7, 9, 0, 9, 0, 6, 8, 0, 6, 9, 2, 2, 0, 6, 2, 2, 0, 6, 8, 1, 0, 6, 8, 0, 5, 8, 9, 9, 2, 2, 2, 2, 2, 0, 5, 9, 0, 7, 0, 0, 9, 6, 9, 0, 6, 1, 1, 0, 7, 0, 2, 6, 2, 0, 6, 1, 1, 1, 0, 2, 6, 2, 2, 2, 0, 1, 9, 0, 0, 8, 1, 1, 0, 5, 8, 0, 7, 1, 1, 0, 9, 6, 2, 6, 6, 8, 0, 6, 9, 0, 8, 0, 7, 8, 0, 0, 5, 9, 2, 0, 5, 2, 5, 0, 0, 6, 2, 0, 8, 0, 6, 9, 2, 0, 6, 9, 0, 6, 9, 0, 5, 0, 9, 2, 2, 0, 2, 2, 5, 2, 0, 6, 0, 5, 0, 1, 1, 6, 0, 9, 5, 0, 2, 2, 0, 6, 8, 0, 7, 8, 1, 0, 8, 0, 6, 0, 8, 0, 8, 0, 2, 2, 0, 6, 0, 5, 1, 0, 6, 9, 0, 8, 0, 2, 6, 2, 0, 5, 1, 0, 6, 9, 0, 6, 9, 2, 0, 8, 6, 1, 0, 5, 0, 5, 8, 8, 0, 6, 2, 0, 5, 1, 1, 0, 6, 9, 0, 2, 2, 0, 0, 6, 1, 1, 0, 2, 0, 8, 0, 7, 9, 2, 0, 2, 6, 0, 5, 0, 6, 8, 1, 0, 6, 8, 0, 9, 0, 2, 2, 0, 8, 5, 0, 5, 9, 0, 7, 1, 0, 6, 9, 0, 2, 0, 6, 8, 1, 0, 1, 7, 0, 6, 8, 0, 7, 0, 5, 8, 1, 0, 9, 2, 0, 1, 0, 2, 0, 5, 9, 0, 7, 8, 1, 0, 6, 2, 0, 1, 0, 8, 0, 6, 0, 6, 0, 5, 1, 0, 6, 9, 0, 9, 0, 2, 6, 2, 0, 6, 6, 8, 0, 6, 1, 1, 0, 2, 0, 5, 1, 0, 9, 0, 6, 2, 8, 1, 0, 6, 2, 8, 0, 8, 1, 0, 2, 0, 6, 8, 0, 0, 9, 0, 6, 1, 1, 0, 6, 9, 0, 2, 6, 2, 2, 0, 8, 7, 1, 1, 0, 0, 6, 9, 2, 0, 6, 8, 0, 2, 0, 5, 2, 2, 2, 0, 1, 6, 1, 5, 5, 2, 0, 1, 2, 0, 6, 1, 9, 2, 0, 6, 9, 0, 5, 1, 1, 1, 0, 2, 2, 0, 8, 5, 2, 0, 0, 6, 1, 1, 1, 0, 2, 8, 0, 6, 0, 2, 6, 2, 0, 2, 5, 1, 6, 6, 1, 0, 5, 1, 9, 0, 2, 0, 6, 2, 8, 0, 6, 8, 1, 0, 0, 2, 5, 1, 1, 1, 1, 1, 0, 6, 2, 2, 1, 0, 2, 5, 2, 2, 0, 1, 6, 1, 1, 2, 0, 5, 0, 1, 6, 1, 1, 1, 1, 0, 2, 0, 0, 6, 1, 1, 0, 7, 1, 1, 0, 6, 8, 0, 6, 8, 0, 8, 5, 0, 8, 1, 0, 6, 9, 0, 9, 2, 0, 9, 0, 5, 1, 0, 8, 6, 6, 0, 5, 2, 2, 0, 5, 0, 5, 0, 9, 0, 2, 6, 2, 0, 2, 6, 0, 5, 1, 1, 0, 5, 2, 0, 8, 0, 8, 0, 8, 5, 0, 6, 9, 0, 6, 8, 0, 5, 9, 0, 9, 2, 2, 0, 9, 5, 1, 0, 5, 9, 0])
    # agent.load("./save/teris1.h5") #Load old ai
    batch_size = 128
    e = 0
    # for e in range(EPISODES):
    while True:
        # print(np.r.seed(), r.seed())
        parent_window.title("AI #" + str(e))

        canvas = Canvas(frame, width=400, height=600)
        canvas.pack()
        frame.pack()
        game = wr.reset_tetris(c=canvas, seed=seed)
        # game = wr.reset_tetris(c=None, seed=seed)
        # seed += 1
        state = wr.output_data()
        state = np.reshape(state, [1, state_size])
        parent_window.update()
        # first = r.randint(0, wr.action_space)
        # print(first)
        # if e < EPISODES:
        #   wr.act_hd(first) #RNG START
        for time in range(1500):
            parent_window.update()

            action = agent.act(state)
            # print(action)
            wr.act_hd(action)
            next_state = wr.output_data()

            v.plot_eyes(next_state, game.input_log)
            reward = game.reward()
            done = not game.active
            reward = reward if not done else -game.total_score * 2
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done or time == 100:
                v.set_last(game.input_log)
                v.add_point(game.total_score)
                v.add_decay(agent.epsilon)

                print("episode: {}/{}, score: {}, e: {:.6}"
                      .format(e, EPISODES, game.total_score, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        canvas.destroy()
        v.plotter()
        if e % 30 == 0:
            agent.save("./save/tetris.h5")
        e += 1


if __name__ == "__main__":
    root = Tk()
    frame = Frame(root, width=1000, height=1000)
    run_ai(root, frame)
    root.mainloop()
