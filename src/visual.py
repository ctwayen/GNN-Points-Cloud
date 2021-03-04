from matplotlib import cm
import networkx as nx
import matplotlib.pyplot as plt

def graph(edges):
    edge = []
    for i in range(len(edges[0])):
        edge.append((edges[0][i], edges[1][i]))
    G = nx.Graph()
    G.add_edges_from(edge)
    return G

def draw_graph(G, pts, axis='x', path=None):
    colors = []
    if axis == 'x':
        base = pts[:, 0]
    elif axis == 'y':
        base = pts[:, 1]
    else:
        base = pts[:, 2]
    colors = cm.coolwarm(base)
    nx.draw(G, pos=nx.spring_layout(G), node_color=colors, node_size = 3, width=0.1)
    if path != None:
        plt.savefig(path)
    plt.show()
def visual(pts, axis ='x', path=None):
    fig = plt.figure(figsize=(9, 6))
    # Create 3D container
    ax = plt.axes(projection = '3d')
    # Visualize 3D scatter plot
    ax.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 1], cmap = cm.coolwarm)
    # Give labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    if path != None:
        fig.savefig(path)