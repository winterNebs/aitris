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

root = tk.Tk()
frame = tk.Frame(root, width=1000, height=1000)
v = vis(root)
while True:
    root.title("AI #" + str(1))

    canvas = tk.Canvas(frame, width=400, height=600)
    canvas.pack()
    frame.pack()
    # game = wr.reset_tetris(c=None, seed=seed)
    wr = wrapper()
    game = wr.reset_tetris(c=canvas, seed=1)
    state = wr.output_data()
    episode_reward = 0

    for step in range(1):
        root.update()
    canvas.destroy()

root.mainloop()