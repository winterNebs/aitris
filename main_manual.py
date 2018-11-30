from tetris import *
import random
from tkinter import *
from tensorflow import keras
from tensorflow import set_random_seed
import numpy as np


root = Tk()
root.title("Tetris")

frame = Frame(root, width=1000, height=1000)
canvas = Canvas(frame, width=400, height=600)
canvas.pack()
frame.pack()

tetris = Tetris(canvas)
# Keyboard Input


def key(event):
    print("pressed", event.keysym)
    converter = ['space', 'Left', 'Right', 'Down', 'Shift_L', 's', 'Up', 'd', 'e', 'q']

    tetris.input_c(converter.index(event.keysym))
    tetris.render()


# Create frame for organizational reasons

# Callback
root.bind("<KeyPress>", key)


# thebiggay = "h, dasr, l, hd, hd, dasl, hd, cw, dasr, hd"
# tetris.run_commands(thebiggay)
# Main Loop


'''tetris.input_log = ['cw', 'dasl', 'hd', 'ccw', 'dasr', 'hd', 'dasl', 'r', 'hd', 'dasl', 'hd', 'r', 'r', 'sd', 'sd',
                    'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd',
                    'sd', 'sd', 'r', 'hd', 'dasr', 'hd', 'ccw', 'r', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd',
                    'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'l', 'l', 'hd', 'cw', 'hd',
                    'r', 'r', 'hd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd', 'sd',
                    'sd', 'sd', 'r', 'r', 'ccw', 'sd', 'sd', 'ccw', 'hd']'''


def loop():
    # tetris.replay()
    root.after(50, loop)


root.after(1, loop)
root.mainloop()

root.mainloop()

