import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import time

from node import Node

def visualize_hex_node_state(node: Node) -> None:
    board_size = node.state.shape[0]
    G = nx.grid_2d_graph(board_size, board_size)
    plt.figure(figsize=(10,10))
    diagonals = []
    for x,y in G:
        x2 = x-1
        y2 = y+1
        if y2 >= board_size or x2 < 0:
            continue
        edge = ((x, y), (x2,y2))
        diagonals.append(edge)
    G.add_edges_from(diagonals)
    pos = {}
    colour_map = []
    theta = -(1/4) * np.pi
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    rotation_matrix = np.array([
        [costheta, -sintheta],
        [sintheta, costheta]
    ])
    for x,y in G:
        coords = (x,y)
        pos[coords] = np.dot(rotation_matrix, (y,-x))
        if node.state[coords] == 1:
            colour_map.append("red")
        elif node.state[coords] == -1:
            colour_map.append("blue")
        else:
            colour_map.append("grey")

    nx.draw(G, pos=pos, 
            node_color=colour_map,
            with_labels=False,
            node_size=600)
    plt.show()


if __name__ == "__main__":
    test_state = np.zeros(shape=(7,7))
    test_state[0, 1] = 1
    test_node = Node(state=test_state)
    visualize_hex_node_state(test_node)