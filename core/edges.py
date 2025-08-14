# core/edges.py
from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple
import itertools
import math

def _euclid(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def add_distance_threshold(G, *, threshold: float, long_interaction_threshold: float = 0, **ctx):
    """
    Garante conectividade baseada na distância de centróides.
    - threshold: corta ou cria arestas com distance <= threshold como 'res-res' ou 'wat-res'
    - long_interaction_threshold > 0: marca conexões entre threshold < d <= long_interaction_threshold como 'long'
      sem remover as curtas

    Requisitos:
      - cada nó precisa ter atributo 'centroid' = (x,y,z)
      - nós de água têm 'kind' = 'water'
    """
    nodes = list(G.nodes(data=True))
    centroids = {n: d.get("centroid") for n, d in nodes if d.get("centroid") is not None}


    for u, v in itertools.combinations(centroids.keys(), 2):
        cu = centroids[u]; cv = centroids[v]
        d = _euclid(cu, cv)
        if d <= threshold and d > 0.0:
            kind = "wat-res" if ("water" in (G.nodes[u].get("kind"), G.nodes[v].get("kind"))) else "res-res"
            G.add_edge(u, v, distance=float(d), kind=kind)

    if long_interaction_threshold and long_interaction_threshold > threshold:
        upper = float(long_interaction_threshold)
        for u, v in itertools.combinations(centroids.keys(), 2):
            cu = centroids[u]; cv = centroids[v]
            d = _euclid(cu, cv)
            if threshold < d <= upper:
                if not G.has_edge(u, v):
                    G.add_edge(u, v, distance=float(d), kind="long")

    return G
