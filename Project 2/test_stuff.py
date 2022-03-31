import networkx as nx
import matplotlib.pyplot as plt
import graphistry
from pylab import *

from node import Node
from visualizer import get_graph

class AnnoteFinder:  # thanks to http://www.scipy.org/Cookbook/Matplotlib/Interactive_Plotting
    """
    callback for matplotlib to visit a node (display an annotation) when points are clicked on.  The
    point which is closest to the click and within xtol and ytol is identified.
    """
    def __init__(self, xdata, ydata, annotes, axis=None, xtol=None, ytol=None):
        self.data = list(zip(xdata, ydata, annotes))
        if xtol is None: xtol = ((max(xdata) - min(xdata))/float(len(xdata)))/2
        if ytol is None: ytol = ((max(ydata) - min(ydata))/float(len(ydata)))/2
        self.xtol = xtol
        self.ytol = ytol
        if axis is None: axis = gca()
        self.axis= axis
        self.drawnAnnotations = {}
        self.links = []

    def __call__(self, event):
        if event.inaxes:
            clickX = event.xdata
            clickY = event.ydata
            # print(dir(event),event.key)
            if self.axis is None or self.axis==event.inaxes:
                annotes = []
                smallest_x_dist = float('inf')
                smallest_y_dist = float('inf')

                for x,y,a in self.data:
                    if abs(clickX-x)<=smallest_x_dist and abs(clickY-y)<=smallest_y_dist :
                        dx, dy = x - clickX, y - clickY
                        annotes.append((dx*dx+dy*dy,x,y, a) )
                        smallest_x_dist=abs(clickX-x)
                        smallest_y_dist=abs(clickY-y)
                        # print(annotes,'annotate')
                    # if  clickX-self.xtol < x < clickX+self.xtol and  clickY-self.ytol < y < clickY+self.ytol :
                    #     dx,dy=x-clickX,y-clickY
                    #     annotes.append((dx*dx+dy*dy,x,y, a) )
                # print(annotes,clickX,clickY,self.xtol,self.ytol )
                if annotes:
                    annotes.sort() # to select the nearest node
                    distance, x, y, annote = annotes[0]
                    self.drawAnnote(event.inaxes, x, y, annote)

    def drawAnnote(self, axis, x, y, annote):
        if (x, y) in self.drawnAnnotations:
            markers = self.drawnAnnotations[(x, y)]
            for m in markers:
                m.set_visible(not m.get_visible())
            self.axis.figure.canvas.draw()
        else:
            t = axis.text(x, y, f"{annote}", )
            m = axis.scatter([x], [y], marker='d', c='r', zorder=100)
            self.drawnAnnotations[(x, y)] = (t, m)
            self.axis.figure.canvas.draw()




# Build your graph
G, pos, colour_map = get_graph(Node(np.zeros(shape=(7,7))))
x, y, annotes = [], [], []
for key in pos:
    d = pos[key]
    annotes.append(key)
    x.append(d[0])
    y.append(d[1])

fig = plt.figure(figsize=(10,10))

nx.draw(G, pos, font_size=6, node_color=colour_map, edge_color='#000000',
                  node_size=600,with_labels=False)


af = AnnoteFinder(x, y, annotes)
connect('button_press_event', af)

show()