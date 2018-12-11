import keras
import numpy as np
import random
from collections import deque
from tetris_wrapper import wrapper
from tkinter import *
from visualizer import vis
r = random.Random(0)
npr = np.random.RandomState(0)
STATE_SIZE = (29,10,1)
TRAIN_SIZE = [1, 29, 10, 1]
class Network:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = .90  # (Short sighted) 0 : 1+ (long term/divergent)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99992
        self.learning_rate = 0.001
        self.model = self.gen_model()

    def gen_model(self):
        model = keras.Sequential()

        '''model.add(keras.layers.Conv2D(32, kernel_size=3,
                         activation='relu',
                         input_shape=self.state_size))
        model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(self.action_size, activation='softmax'))'''
        model.add(keras.layers.Conv2D(32, (29, 1), padding='same', activation='relu', input_shape=self.state_size))
        model.add(keras.layers.Conv2D(32, (1, 10), padding='same',  activation='relu'))
        #model.add(keras.layers.Conv2D(64, (2, 10),padding='same',  activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(29, 1)))
        #model.add(keras.layers.Dropout(0.25))
        '''
        model.add(keras.layers.Conv2D(64, (2, 10), padding='same', activation='relu', input_shape=self.state_size))
        model.add(keras.layers.Conv2D(64, (2, 10), activation='relu'))
        # model.add(keras.layers.MaxPooling2D(pool_size=(12, 5)))
        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Conv2D(64, (2, 10), padding='same', activation='relu', input_shape=self.state_size))
        model.add(keras.layers.Conv2D(64, (1, 10), activation='relu'))
        # model.add(keras.layers.MaxPooling2D(pool_size=(12, 5)))
        model.add(keras.layers.Dropout(0.25))
        '''
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        #model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(self.action_size, activation='softmax'))
        model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if npr.rand() <= self.epsilon:
            return npr.randint(low=0, high=self.action_size)
        act_values = self.model.predict(state)
        #print(act_values)
        return np.argmax(act_values[0])  # returns action

    def relearn(self, batch_size):
        minibatch = r.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            #print(self.epsilon, self.epsilon * self.epsilon_decay)
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.epsilon = self.epsilon_min
        #self.epsilon = 0.35

    def save(self, name):
        self.model.save_weights(name)


def train(agent, input_log):
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
    state = np.reshape(state, TRAIN_SIZE)

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
        next_state = np.reshape(next_state, TRAIN_SIZE)
        agent.remember(state, action, reward, next_state, done)
        state = next_state

    agent.relearn(int(len(input_log)/4))
    root.destroy()


EPISODES = 3000


def run_ai(parent_window, frame):
    v = vis(root)
    seed = 0
    wr = wrapper()
    action_size = wr.action_space

    agent = Network(STATE_SIZE, action_size)
    # train(agent, [6, 9, 0, 5, 8, 0, 9, 2, 0, 9, 0, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 0, 8, 0, 5, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 0, 6, 0, 2, 2, 0, 5, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 0, 2, 2, 2, 0, 0, 6, 2, 0, 9, 0, 1, 0, 9, 6, 1, 0, 6, 8, 0, 5, 9, 2, 0, 8, 1, 0, 5, 0, 6, 2, 0, 9, 0, 8, 0, 8, 1, 0, 9, 6, 0, 5, 0, 8, 0, 6, 9, 0, 2, 2, 0, 8, 1, 0, 2, 2, 0, 6, 0, 7, 1, 0, 1, 0, 1, 0, 7, 0, 7, 8, 1, 0, 6, 8, 0, 5, 1, 0, 9, 0, 2, 0, 6, 9, 0, 6, 9, 2, 0, 8, 1, 0, 5, 2, 0, 2, 6, 2, 0, 6, 8, 0, 1, 0, 0, 9, 2, 0, 0, 2, 2, 6, 0, 5, 1, 0, 6, 8, 0, 6, 9, 0, 6, 8, 1, 0, 5, 9, 0, 2, 2, 5, 2, 0, 2, 0, 5, 1, 0, 0, 9, 2, 0, 6, 9, 0, 7, 9, 2, 0, 5, 8, 1, 0, 6, 8, 0, 6, 9, 0, 8, 1, 1, 0, 0, 0, 5, 1, 0, 7, 0, 6, 8, 0, 7, 9, 0, 9, 0, 6, 8, 0, 6, 9, 2, 2, 0, 6, 2, 2, 0, 6, 8, 1, 0, 6, 8, 0, 5, 8, 9, 9, 2, 2, 2, 2, 2, 0, 5, 9, 0, 7, 0, 0, 9, 6, 9, 0, 6, 1, 1, 0, 7, 0, 2, 6, 2, 0, 6, 1, 1, 1, 0, 2, 6, 2, 2, 2, 0, 1, 9, 0, 0, 8, 1, 1, 0, 5, 8, 0, 7, 1, 1, 0, 9, 6, 2, 6, 6, 8, 0, 6, 9, 0, 8, 0, 7, 8, 0, 0, 5, 9, 2, 0, 5, 2, 5, 0, 0, 6, 2, 0, 8, 0, 6, 9, 2, 0, 6, 9, 0, 6, 9, 0, 5, 0, 9, 2, 2, 0, 2, 2, 5, 2, 0, 6, 0, 5, 0, 1, 1, 6, 0, 9, 5, 0, 2, 2, 0, 6, 8, 0, 7, 8, 1, 0, 8, 0, 6, 0, 8, 0, 8, 0, 2, 2, 0, 6, 0, 5, 1, 0, 6, 9, 0, 8, 0, 2, 6, 2, 0, 5, 1, 0, 6, 9, 0, 6, 9, 2, 0, 8, 6, 1, 0, 5, 0, 5, 8, 8, 0, 6, 2, 0, 5, 1, 1, 0, 6, 9, 0, 2, 2, 0, 0, 6, 1, 1, 0, 2, 0, 8, 0, 7, 9, 2, 0, 2, 6, 0, 5, 0, 6, 8, 1, 0, 6, 8, 0, 9, 0, 2, 2, 0, 8, 5, 0, 5, 9, 0, 7, 1, 0, 6, 9, 0, 2, 0, 6, 8, 1, 0, 1, 7, 0, 6, 8, 0, 7, 0, 5, 8, 1, 0, 9, 2, 0, 1, 0, 2, 0, 5, 9, 0, 7, 8, 1, 0, 6, 2, 0, 1, 0, 8, 0, 6, 0, 6, 0, 5, 1, 0, 6, 9, 0, 9, 0, 2, 6, 2, 0, 6, 6, 8, 0, 6, 1, 1, 0, 2, 0, 5, 1, 0, 9, 0, 6, 2, 8, 1, 0, 6, 2, 8, 0, 8, 1, 0, 2, 0, 6, 8, 0, 0, 9, 0, 6, 1, 1, 0, 6, 9, 0, 2, 6, 2, 2, 0, 8, 7, 1, 1, 0, 0, 6, 9, 2, 0, 6, 8, 0, 2, 0, 5, 2, 2, 2, 0, 1, 6, 1, 5, 5, 2, 0, 1, 2, 0, 6, 1, 9, 2, 0, 6, 9, 0, 5, 1, 1, 1, 0, 2, 2, 0, 8, 5, 2, 0, 0, 6, 1, 1, 1, 0, 2, 8, 0, 6, 0, 2, 6, 2, 0, 2, 5, 1, 6, 6, 1, 0, 5, 1, 9, 0, 2, 0, 6, 2, 8, 0, 6, 8, 1, 0, 0, 2, 5, 1, 1, 1, 1, 1, 0, 6, 2, 2, 1, 0, 2, 5, 2, 2, 0, 1, 6, 1, 1, 2, 0, 5, 0, 1, 6, 1, 1, 1, 1, 0, 2, 0, 0, 6, 1, 1, 0, 7, 1, 1, 0, 6, 8, 0, 6, 8, 0, 8, 5, 0, 8, 1, 0, 6, 9, 0, 9, 2, 0, 9, 0, 5, 1, 0, 8, 6, 6, 0, 5, 2, 2, 0, 5, 0, 5, 0, 9, 0, 2, 6, 2, 0, 2, 6, 0, 5, 1, 1, 0, 5, 2, 0, 8, 0, 8, 0, 8, 5, 0, 6, 9, 0, 6, 8, 0, 5, 9, 0, 9, 2, 2, 0, 9, 5, 1, 0, 5, 9, 0])
    #agent.load("./save/teris1.h5") #Load old ai
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
        # seed += 1
        state = wr.output_data()
        state = np.reshape(state, TRAIN_SIZE)
        parent_window.update()
        first = r.randint(0, 34)
        # print(first)
        if e > EPISODES:
            wr.act_hd(first) #RNG START
        for time in range(600):
            parent_window.update()

            action = agent.act(state)
            # print(action)
            wr.act_hd(action)
            next_state = wr.output_data()

            v.set_eyes(next_state)
            v.plot_eyes()
            reward = game.reward()
            done = not game.active
            reward = reward if not done else -game.total_score * 2
            next_state = np.reshape(next_state, TRAIN_SIZE)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done or time == 598:

                v.add_point(game.total_score)
                v.add_decay(agent.epsilon)

                print("episode: {}/{}, score: {}, e: {:.6}"
                      .format(e, EPISODES, game.total_score, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.relearn(batch_size)
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
