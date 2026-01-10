import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# def draw_graph(graph: nx.Graph, show: bool = True, save_path: Optional[str] = None):
#     """Visualize and optionally save a graph."""
#     if not show and not save_path:
#         raise ValueError("Either 'show' or 'save_path' must be set to True.")

#     node_colors = [graph.nodes[node].get("chain_id", "gray") for node in graph.nodes]
#     nx.draw(graph, with_labels=True, node_color=node_colors, node_size=50, font_size=6)

#     if show:
#         plt.show()
#     if save_path:
#         Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(save_path)
#         plt.clf()
#         print(f"Graph plot saved to {save_path}")
