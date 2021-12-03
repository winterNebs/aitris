
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Basically just render graphs, nothing too crazy here
class vis:
    def __init__(self, root):
        self.last_data = [1]
        self.points = []
        self.decay = []
        self.plotting = False
        self.fig = Figure()
        gs = self.fig.add_gridspec(2, 3)
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.ax3 = self.fig.add_subplot(gs[:, 1])
        self.ax4 = self.fig.add_subplot(gs[0, 2])

        self.graph = FigureCanvasTkAgg(self.fig, master=root)
        self.graph.get_tk_widget().pack(side="left", fill='both', expand=True)

    def set_last(self, data):
        self.last_data += data

    def add_point(self, point):
        self.points.append(point)

    def add_decay(self, point):
        self.decay.append(point)

    def plot_eyes(self, eyes, input):
        self.ax3.cla()
        self.ax3.grid()
        self.ax3.set_title("What the AI sees")

        self.ax3.imshow(eyes)

        self.graph.draw()

    def plotter(self):
        self.ax1.cla()
        self.ax1.grid()
        self.ax1.set_title("Total Score")
        dpts = self.points
        self.ax1.scatter(range(len(dpts)), dpts)

        self.ax2.cla()
        self.ax2.grid()
        self.ax2.set_title("Epsilon")
        dpts = self.decay
        self.ax2.plot(dpts, marker='o', color='blue')

        self.graph.draw()


