import csv
import math

import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go


def get_nx_graph(nodes, edges, rm_iso=False):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=int(math.log10(edge[2] + 1)) + 1)
    if rm_iso:
        G.remove_nodes_from(nx.isolates(G))
    return G


def paint_by_networkx(nodes, edges):
    G = get_nx_graph(nodes, edges)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw(G, node_color='red', width=weights)
    plt.show()


def paint_by_plotly(nodes, edges, color_marks=None):
    G = get_nx_graph(nodes, edges, rm_iso=True)

    pos = nx.spring_layout(G, k=0.5, iterations=100)
    for n, p in pos.items():
        G.nodes[n]['pos'] = p

    edge_traces = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_trace = go.Scatter(
            x=(x0, x1), y=(y0, y1),
            line=dict(width=G[edge[0]][edge[1]]['weight'], color='#888'),
            hoverinfo='none',
            mode='lines')
        edge_traces.append(edge_trace)

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_text.append('# of connections: ' + str(len(adjacencies[1])))

    if not color_marks:
        color_marks = [0] * len(nodes)
    node_trace.marker.color = color_marks
    node_trace.text = node_text

    fig = go.Figure(data=edge_traces,
                    layout=go.Layout(
                        title='<br>Network graph made with Python',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.add_trace(node_trace)
    return fig

"""
dates = ["2023-10-13"]
records = []
limited = 100
for date in dates:
    path = get_data_path(date)
    with open('./data/' + date + '.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            if limited <= 0:
                break
            if row:
                records.append((row[0], row[1], int(float(row[2])) + 1))
                limited -= 1

nodes = []
for row in records:
    if row[0] not in nodes:
        nodes.append(row[0])
    if row[1] not in nodes:
        nodes.append(row[1])
edges = records
paint_by_plotly(nodes, edges)
"""

