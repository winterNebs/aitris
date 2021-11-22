import keras
import numpy as np
import random
from collections import deque
from tetris_wrapper import wrapper
from tkinter import *
from visualizer import vis
r = random.Random(0)
npr = np.random.RandomState(0)

# train network using input_log
def train(agent, input_log):
    state_size = 290
    root = Tk()
    root.title("Tetris")

    frame = Frame(root, width=1000, height=1000)
    canvas = Canvas(frame, width=400, height=600)
    canvas.pack()
    frame.pack()

    wr = wrapper(c=canvas)
    game = wrapper.tetris
    # game = Tetris()
    state = wr.output_data()
    state = np.reshape(state, [1, state_size])

    for command in input_log:
        #env.render()
        root.update()
        #cmd = ['hd', 'l', 'r', 'sd', 'h', 'ccw', 'cw', '180', 'dasr', 'dasl']
        action = command
        #cmd.index(command)
        wr.act_hd(action)
        next_state = wr.output_data()
        reward = game.reward()
        done = not game.active
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state

    agent.replay(int(len(input_log)/4))
    root.destroy()


EPISODES = 6000


def deep_q(parent_window, frame):
    v = vis(root)
    seed = 0
    wr = wrapper()
    state_size = 290
    action_size = wr.action_space

    agent = Network(state_size, action_size)
    #agent.load("./save/tetris.h5") #Load old ai
    batch_size = 32
    e = 0
    # for e in range(EPISODES):
    while True:
        # print(np.r.seed(), r.seed())
        parent_window.title("AI #" + str(e))

        canvas = Canvas(frame, width=400, height=600)
        canvas.pack()
        frame.pack()
        game = wr.reset_tetris(c=canvas, seed=seed)
        #game = wr.reset_tetris(c=None, seed=seed)
        #seed += 1
        state = wr.output_data()
        state = np.reshape(state, [1, state_size])
        parent_window.update()
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
