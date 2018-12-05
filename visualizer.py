from tkinter import *
from random import randint

# these two imports are important
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
import threading

class vis:
    def __init__(self, root):
        self.points = []
        self.decay = []
        self.eyes_emoji = []
        self.plotting = False
        self.fig = Figure()
        gs = self.fig.add_gridspec(2, 2)
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.ax3 = self.fig.add_subplot(gs[:, 1])

        self.graph = FigureCanvasTkAgg(self.fig, master=root)
        self.graph.get_tk_widget().pack(side="left", fill='both', expand=True)

    def add_point(self, point):
        self.points.append(point)

    def add_decay(self, point):
        self.decay.append(point)

    def set_eyes(self, data):
        self.eyes_emoji = data

    def plot_eyes(self):
        self.ax3.cla()
        self.ax3.grid()
        self.ax3.set_title("What the AI sees")

        self.ax3.imshow(self.eyes_emoji)

        self.graph.draw()

    def plotter(self):
        self.ax1.cla()
        self.ax1.grid()
        self.ax1.set_title("Total Score")
        #self.ax1.set_xlabel("AI Number")
        #self.ax1.set_ylabel("Total Score")
        dpts = self.points
        self.ax1.scatter(range(len(dpts)), dpts)

        self.ax2.cla()
        self.ax2.grid()
        self.ax2.set_title("Epsilon")
        #self.ax2.set_xlabel("AI Number")
        #self.ax2.set_ylabel("Epsilon")
        dpts = self.decay
        self.ax2.plot(dpts, marker='o', color='blue')

        self.graph.draw()


