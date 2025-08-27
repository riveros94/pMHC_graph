from core.tracking import save
from Bio.PDB import *
from collections import defaultdict
import numpy as np
from itertools import product
from Bio.PDB.vectors import Vector
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
import networkx as nx
from graphein.protein.edges.distance import compute_distmat
from os import path
import pandas as pd
from collections import Counter
from typing import Any, FrozenSet, Tuple, List, Optional, Union, Dict, Set
import itertools
import logging
import matplotlib.pyplot as plt
from utils.cutils.combinations_filter import filtered_combinations
from SERD_Addon.classes import StructureSERD
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import json
import math

log = logging.getLogger("CRSProtein")

def save_pdb_with_spheres(atomic_data, selected_residues_data, pdb_filename):
    """
    Salva o arquivo PDB com esferas para os resíduos selecionados.

    Parameters
    ----------
    atomic_data : numpy.ndarray
        Dados atômicos da estrutura.
    selected_residues_data : list
        Lista com os dados dos resíduos selecionados, incluindo coordenadas e profundidade.
    pdb_filename : str
        Nome do arquivo PDB de saída.
    """

    
    with open(pdb_filename, 'w') as f:
        # Adiciona os átomos originais
        for row in atomic_data:
            atom_line = "ATOM  {:5d}  {:<4} {:<3} {:<1} {:>4} {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}\n".format(
                int(row[0]), row[3], row[2], row[1], int(row[4]), row[5], row[6], row[7], row[8], row[9]
            )
            f.write(atom_line)
        
        # Adiciona as esferas para os resíduos selecionados
        for residue in selected_residues_data:
            residue_number = residue["ResidueNumber"]
            chain = residue["Chain"]
            coordinates = residue["Coordinates"]
            radius = 1.5  # Definindo um raio para a es fera

            sphere_line = "HETATM{:5d}  SPC {:<3} {:<1} {:>4}   {:>8.3f}{:>8.3f}{:>8.3f}  1.00  0.00           SPH\n".format(
                residue_number * 100, "SPC", chain, residue_number, coordinates[0], coordinates[1], coordinates[2]
            )
            f.write(sphere_line)

    print(f"PDB com esferas salvo em {pdb_filename}")

def check_identity(node_pair: Tuple):
    '''
    Takes a tuple of nodes in the following format: 'A:PRO:57', 'A:THR:178')
    '''
    
    if isinstance(node_pair[-1], tuple):
        raise Exception(f"The node of second graph is a tuple {node_pair}")
    
    second_graph_node = node_pair.pop(-1) 
    node_ref = []
    
    while True:
        if isinstance(node_pair[-1], tuple):
            node_pair = node_pair[0]
        elif isinstance(node_pair[-1], str):
            node_ref = node_pair[-1]
            break
        else:
            raise Exception(f"Unexpected type of node. Node: {node_pair[-1]}, type: {type(node_pair[-1])}")
        
    if node_ref.split(":")[1] == node_pair.split(':')[1]:
        return True
    else:
        return False
    
def check_cross_positions(node_pair_pair: Tuple[Tuple, Tuple]):
    """Check if there are cross position of residues between nodes

    Args:
        node_pair_pair (Tuple[Tuple]): A tuple that contain tuple of nodes

    Returns:
        bool: Return true if you don't have cross position between residues
    """
    
    nodeA, nodeB = node_pair_pair[0], node_pair_pair[1]
    setA, setB = set(nodeA), set(nodeB)
    lenA, lenB = len(setA), len(setB)
    not_cross = all(nodeA[k] != nodeB[k] for k in range(len(nodeA))) 
    repeated = (lenA == 1 and lenB != 1) or (lenB == 1 and lenA != 1)
    # permutation = setA == setB
    
    return not_cross  and not repeated

def convert_edges_to_residues(edges: Set[FrozenSet], maps: Dict, debug: bool = False) -> Tuple[List, List]:
    """Convert the edges that contains tuple of indices to tuple of residues

    Args:
        edges (List[Tuple]): A list that contains tuple of edges that are made of tuples of indices
        maps (Dict): A map that relates the indice to residue

    Returns:
        convert_edge (List[Tuple]): Return edges converted to residues notation
    """
    # for edge in edges_base:
    #     list_edge = list(edge)
    #     edges_indices_base.append(tuple((possible_nodes_map[list_edge[0]], possible_nodes_map[list_edge[1]])))

    original_edges = []
    edges_indices = []
    converted_edges = []
    residue_maps_unique = maps["residue_maps_unique"]
    possible_nodes_map = maps["possible_nodes"] 
    for edge in edges:
        edge_list = list(edge)
        node1, node2 = possible_nodes_map[edge_list[0]], possible_nodes_map[edge_list[1]]
        converted_node1 = tuple(f"{residue_maps_unique[idx][0]}:{residue_maps_unique[idx][2]}:{residue_maps_unique[idx][1]}" for idx in node1)
        converted_node2 = tuple(f"{residue_maps_unique[idx][0]}:{residue_maps_unique[idx][2]}:{residue_maps_unique[idx][1]}" for idx in node2)

        converted_node1_indice = tuple(idx for idx in node1)
        converted_node2_indice = tuple(idx for idx in node2)
        if set(converted_node1) != set(converted_node2) and check_cross_positions((converted_node1, converted_node2)):
            original_edges.append(edge)
            edges_indices.append((converted_node1_indice, converted_node2_indice))
            converted_edges.append((converted_node1, converted_node2))
        else:
            if debug: log.debug(f'Invalid edge: {edge}, {converted_node1}:{converted_node2}')
    return original_edges, edges_indices, converted_edges
    
def check_multiple_chains(node: Tuple, residue_maps_unique: Dict):
    """Check if at least on node is from different chain

    Args:
        node (tuple): Tuple of residues that together makes a node
        residue_maps_unique (Dict): A dictionary of all residues

    Returns:
        boolean: Return true if there's a different chain in node
    """
    
    chains = set([residue_maps_unique[node_indice][0] for node_indice in node])
    if len(chains) > 1:
        return False
    else:
        return True

def filter_maps_by_nodes(data: dict, 
                        matrices_dict: dict,
                        distance_threshold: float = 10.0, 
                    ) -> Tuple[Dict, Dict]:

    logger = logging.getLogger("association.filter_maps_by_nodes")
    
    contact_maps = data["contact_maps"]
    rsa_maps = data["rsa_maps"]
    residue_maps = data["residue_maps"]
    nodes_graphs = data["nodes_graphs"]

    
    maps = {"full_residue_maps": [], "residue_maps_unique": {}}
    pruned_contact_maps = []
    thresholded_contact_maps = []
    thresholded_rsa_maps = []
    
    for contact_map, rsa_map, residue_map, nodes in zip(
            contact_maps, rsa_maps, residue_maps, nodes_graphs):
         
        indices = []
        
        for node in nodes:
            parts = node.split(":")
            if len(parts) != 3:
                print(f"Node '{node}' does not have three parts separated by ':'")
                continue

            chain, res_name, res_num_str = parts
            key = (chain, res_num_str, res_name)
            if key in residue_map:
                indices.append(residue_map[key])

        pruned_map = contact_map[np.ix_(indices, indices)]
        np.fill_diagonal(pruned_map, np.nan)
        pruned_contact_maps.append(pruned_map)
        
        thresh_map = pruned_map.copy()
        thresh_map[thresh_map >= distance_threshold] = np.nan
        thresholded_contact_maps.append(thresh_map)
        
        thresholded_rsa_maps.append(rsa_map)
         
        full_res_map = {}
        for i, node in enumerate(nodes):
            parts = node.split(":")
            if len(parts) != 3:
                logger.warning(f"Node '{node}' does not have three parts; skipping for full residue map")
                continue

            chain, res_name, res_num_str = parts
            full_res_map[(chain, res_num_str, res_name)] = i
        maps["full_residue_maps"].append(full_res_map)
    
    if matrices_dict is not None:
        matrices_dict["pruned_contact_maps"] = pruned_contact_maps
        matrices_dict["thresholded_contact_maps"] = thresholded_contact_maps
        matrices_dict["thresholded_rsa_maps"] = thresholded_rsa_maps
        matrices_dict["thresholded_depth_maps"] = None
    
    return matrices_dict, maps

def indices_graphs(nodes_lists: List[List]) -> List[Tuple[int, int]]:
    """Make a list that contains indices that indicates the position of each protein in graph

    Args:
        nodes_list (List): A list of protein's resdiues. Each List has their own residues.

    Returns:
        ranges (List[Tuple]): A list of indicesthat indicates the position of each protein in matrix
    """
    
    current = 0
    ranges = []
    for nodes in nodes_lists:
        length = len(nodes)
        ranges.append((current, current + length))
        current += length
    return ranges
    
def create_similarity_matrix(nodes_graphs: list,
                             metadata: dict,
                             residues_factors,
                             similarity_cutoff: float = 0.95,
                             mode="dictionary"):

    total_length = metadata["total_length"]
    ranges_graph = metadata["ranges_graph"]
    
    matrix = np.zeros((total_length, total_length))
    
    for i in range(len(nodes_graphs)):
        for j in range(i + 1, len(nodes_graphs)):
            residuesA = residues_factors[i]
            residuesB = residues_factors[j]
            
            if mode == "dictionary":
                keysA = sorted(residuesA.keys())
                keysB = sorted(residuesB.keys())

                A = np.array([residuesA[k] for k in keysA])
                B = np.array([residuesB[k] for k in keysB])

                similarities = cosine_similarity(A, B)

            elif mode == "1d":
                # def normalize(arr):
                #     return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

                residuesA = residuesA.reshape(-1, 1)
                residuesB = residuesB.reshape(1, -1)

                similarities = 1 - np.abs(residuesA - residuesB)
            else:
                return None
            startA, endA = ranges_graph[i]
            startB, endB = ranges_graph[j]

            matrix[startA:endA, startB:endB] = similarities
            matrix[startB:endB, startA:endA] = similarities.T
    
    matrix[matrix < similarity_cutoff] = 0
    matrix[matrix >= similarity_cutoff] = 1
    
    return matrix

def create_residues_factors(graphs: List, factors_path: str):
    
    read_emb = pd.read_csv(factors_path)
    residue_factors = {}
    
    for i, graph in enumerate(graphs):
        nodes = sorted(list(graph.nodes()))
        residue_factors[i] = {node: calculate_atchley_average(node, read_emb) for node in nodes}     
    return residue_factors

def value_to_class(value, n_divisions, threshold, inverse=False, upper_bound = 100):

    if not inverse:
        if value <= 0 or value > threshold:
            return None
        span = threshold
        rel = value
    else:
        if value < threshold:
            return None

        span = upper_bound - threshold
        rel = value - threshold

    length_division = span / n_divisions
    class_name = math.ceil(rel / length_division) 
     
    return class_name

def create_classes_bin(lenght, n_bins): 
    length_division = lenght / n_bins

    return  {
            str(n): ((n-1)*length_division, n*length_division)
            for n in range(n_bins)
        }
 
def find_class(dictionary, value):
    for key in dictionary.keys():
        range_tuple = dictionary[key]
        if range_tuple[0] < value <= range_tuple[1]:
            return key
    else:
        return None

def residue_to_tuple(res):
    res_split = res.split(":")

    return (res_split[0], res_split[2], res_split[1])

def find_triads(graph_data, classes, config, checks):
    G = graph_data["graph"]
    depth = graph_data["residue_depth"]
    rsa = graph_data["rsa"]
    contact_map = graph_data["contact_map"]
    residue_map = graph_data["residue_map_all"]

    triads = {}
    
    if "residues" in classes.keys():
        residue_classes = {
            res: class_name
            for class_name, residues in classes["residues"].items()
            for res in residues
        }
    else:
        residue_classes = None
 
    rsa_classes = classes["rsa"] if "rsa" in classes.keys() else None 
    depth_classes = classes["depth"] if "depth" in classes.keys() else None
    distance_classes = classes["distance"] if "distance" in classes.keys() else None

    n_nodes = len(G.nodes())
    n_edges = len(G.edges())
    
    for center in G.nodes():
        neighbors = {n for n in G.neighbors(center) if n != center}
        for u, w in combinations(neighbors, 2):
            outer_sorted = tuple(sorted([u, w]))
            u_split, center_split, w_split = outer_sorted[0].split(":"), center.split(":"), outer_sorted[1].split(":")
            u_res, center_res, w_res = u_split[1], center_split[1], w_split[1]
            u_tuple, center_tuple, w_tuple = residue_to_tuple(outer_sorted[0]), residue_to_tuple(center), residue_to_tuple(outer_sorted[1])
            u_index, center_index, w_index = residue_map[u_tuple], residue_map[center_tuple], residue_map[w_tuple]
            
            if residue_classes is not None:
                u_res_class, center_res_class, w_res_class = residue_classes[u_res], residue_classes[center_res], residue_classes[w_res] 
            else:
                u_res_class, center_res_class, w_res_class = u_res, center_res, w_res
            
            
            u_resChain = str(u_split[2]) + u_split[0] 
            center_resChain = str(center_split[2]) + center_split[0] 
            w_resChain = str(w_split[2]) + w_split[0] 

            d1 = contact_map[u_index, center_index]
            d2 = contact_map[u_index, w_index]
            d3 = contact_map[center_index, w_index]

            rsa1 = rsa[outer_sorted[0]]*100
            rsa2 = rsa[center]*100
            rsa3 = rsa[outer_sorted[1]]*100

            if checks["depth"]:
                depth1 = depth.loc[depth["ResNumberChain"] == u_resChain]["ResidueDepth"].values[0]
                depth2 = depth.loc[depth["ResNumberChain"] == center_resChain]["ResidueDepth"].values[0]
                depth3 = depth.loc[depth["ResNumberChain"] == w_resChain]["ResidueDepth"].values[0]

                if depth_classes is not None:
                    depth1_class = find_class(depth_classes, depth1)
                    depth2_class = find_class(depth_classes, depth2)
                    depth3_class = find_class(depth_classes, depth3)
                else:
                    depth1_class = value_to_class(depth1, config["depth_bins"], config["depth_filter"])
                    depth2_class = value_to_class(depth2, config["depth_bins"], config["depth_filter"])
                    depth3_class = value_to_class(depth3, config["depth_bins"], config["depth_filter"])
            else:
                depth1_class, depth2_class, depth3_class = 0, 0, 0
                
            if checks["rsa"]:
                if rsa_classes is not None:
                    rsa1_class = find_class(rsa_classes, rsa1)
                    rsa2_class = find_class(rsa_classes, rsa2)
                    rsa3_class = find_class(rsa_classes, rsa3)
                else:
                    rsa1_class = value_to_class(rsa1, config["rsa_bins"], config["rsa_filter"]*100, inverse=True)
                    rsa2_class = value_to_class(rsa2, config["rsa_bins"], config["rsa_filter"]*100, inverse=True)
                    rsa3_class = value_to_class(rsa2, config["rsa_bins"], config["rsa_filter"]*100, inverse=True)
            else:
                rsa1_class, rsa2_class, rsa3_class = 0, 0, 0
                
            if distance_classes is not None:
                d1_class = find_class(distance_classes, d1) 
                d2_class = find_class(distance_classes, d2)
                d3_class = find_class(distance_classes, d3)
            else:
                d1_class = value_to_class(d1, config["distance_bins"], config["centroid_threshold"])
                d2_class = value_to_class(d2, 2*config["distance_bins"], 2*config["centroid_threshold"])
                d3_class = value_to_class(d3, config["distance_bins"], config["centroid_threshold"] )
                
            full_describer = (depth1_class, depth2_class, depth3_class, rsa1_class, rsa2_class, rsa3_class, d1_class, d2_class, d3_class) 
                
            if None not in full_describer:
                triad = (u_res_class, center_res_class, w_res_class, *full_describer) 
                
                if triad not in triads.keys():
                    triads[triad] = {
                        "count": 1,
                        "triads_full": [(outer_sorted[0], center, outer_sorted[1], *full_describer)]
                    }

                else: 
                    triads[triad]["count"] += 1
                    triads[triad]["triads_full"].append((outer_sorted[0], center, outer_sorted[1], *full_describer))
            # else:
                # logging.debug(f"None Found: ({outer_sorted[0], center, outer_sorted[1], d1, d2, d3} | {full_describer}")
    n_triad = 0
    counters = {}
    for triad in triads.keys():
        n_triad += triads[triad]["count"]
        if triads[triad]["count"] not in counters.keys():
            counters[triads[triad]["count"]] = 1
        else:
            counters[triads[triad]["count"]] += 1
    
    logging.info(f"N Nodes: {n_nodes} | N Edges: {n_edges} | N Triad: {n_triad} | Unique Triad: {len(triads.keys())}")
    logging.debug(f"Counters: {counters}")

    return triads

def create_residues_classes(path, residues_similarity_cutoff):

    atchley_factors = pd.read_csv(path, index_col = 0)

    sim = cosine_similarity(atchley_factors.values)
    aas = [convert_1aa3aa(aa) for aa in atchley_factors.index.tolist()]
    
    parent = {aa: aa for aa in aas}
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    for i in range(len(aas)):
        for j in range(i+1, len(aas)):
            if sim[i, j] >= residues_similarity_cutoff:
                union(aas[i], aas[j])
    
    root_to_class = {}

    residue_to_class = {}
    class_id = 1 
    for aa in aas:
        root = find(aa)
        if root not in root_to_class:
            root_to_class[root] = f"C{class_id}"
            class_id += 1 
        residue_to_class[aa] = root_to_class[root]

    return residue_to_class

def cross_protein_triads(triads_per_protein):
    """
    Parameters
    ----------
    triads_per_protein : list[dict]
        Each dict is the `triads` object you already create for one protein.
        Keys  -> token tuples, e.g. ('C1','C2','C4',3,10,3)
        Values -> {"count": int, "triads_full": [triplet tuples …]}

    Returns
    -------
    dict
        token -> list of combinations, where each combination contains
                 exactly one element from each protein’s `triads_full`.
                 Example length‑3 combination:
                 (
                   ('A:ALA:15', 'A:LYS:17', 'A:LEU:20', 1, 5, 10),  # protein‑1
                   ('B:ALA:33', 'B:LYS:35', 'B:LEU:39', 1, 5, 10),  # protein‑2
                   ('C:ALA:7',  'C:LYS:12', 'C:LEU:19', 1, 5, 10)   # protein‑3
                 )
    """
    # Tokens present in *all* proteins
    common_tokens = set.intersection(*(set(t.keys()) for t in triads_per_protein))
    cross_combos = {}
    for token in common_tokens:
        triad_lists = [prot_triads[token]["triads_full"] for prot_triads in triads_per_protein]
        cross_combos[token] = list(itertools.product(*triad_lists))  # full Cartesian product
     
    return cross_combos

def tokens_compatible(t1, t2):
    """
    Each token is (res1, res2, res3, dist12, dist13, dist23).
    Returns True if exactly two residues are shared, both central residues
    (t1[1] and t2[1]) are among those two, and their inter-residue distance matches.
    """
    (A, B, C, d1, d2, d3) = t1
    (X, Y, Z, e1, e2, e3) = t2

    cnt1 = Counter([A, B, C])
    cnt2 = Counter([X, Y, Z])
    shared = list((cnt1 & cnt2).elements())
    if len(shared) != 2:
        return False
    if B not in shared or Y not in shared:
        return False

    # distances must match on the shared pair
    # def pair_dist(a, b, d1, d2, d3):
    #     key = tuple(sorted((a, b)))
    #     return { tuple(sorted((A, B))): d1,
    #              tuple(sorted((A, C))): d2,
    #              tuple(sorted((B, C))): d3 }.get(key)

    def build_map(a, b, c, d1, d2, d3):
        return {
            frozenset((a, b)): d1,
            frozenset((a, c)): d2,
            frozenset((b, c)): d3,
        }

    map1 = build_map(A, B, C, d1, d2, d3)
    map2 = build_map(X, Y, Z, e1, e2, e3)

    key = frozenset(shared)
    return map1.get(key) == map2.get(key)

    # shared_elems = list(shared.elements())
    # key = tuple(sorted(shared_elems))
    # return pair_dist(A, B, d1, d2, d3) == pair_dist(X, Y, e1, e2, e3)

def build_token_compatibility(cross_combos):
    """
    Pattern‐indexed compatibility:

      • We generate 8 “wildcard” patterns per token:
          (resA, resB, *, dAB, *, *), (resB, resA, *, dAB, *, *),
          (*, resA, resB, *, *, dAB), (*, resB, resA, *, *, dAB),
          (resC, resB, *, dBC, *, *), ... etc.

      • We invert them into a pattern→tokens index in O(t·p).

      • For each token, we union its 8 patterns’ token‐lists to get a small
        candidate set, then run your exact `tokens_compatible` only on those.
    """
    tokens = list(cross_combos.keys())
    pattern_map    = defaultdict(set)
    token_patterns = {}

    for t in tokens:
        A, B, C, _, _, _, _, _, _, d1, _, d3 = t
        pats = [
            (A,    B,    None,  None,   None,   None,   None,   None,   None,   d1, None,   None),
            (B,    A,    None,  None,   None,   None,   None,   None,   None,   d1, None,   None),
            (None, A,    B,     None,   None,   None,   None,   None,   None,   None,   None,   d1),
            (None, B,    A,     None,   None,   None,   None,   None,   None,   None,   None,   d1),

            (C,    B,    None,  None,    None,  None,   None,   None,   None,   d3, None, None),
            (B,    C,    None,  None,    None,  None,   None,   None,   None,   d3, None, None),
            (None, B,    C,     None,    None,  None,   None,   None,   None,   None, None, d3),
            (None, C,    B,     None,    None,  None,   None,   None,   None,   None, None, d3),
        ]
        token_patterns[t] = pats
        for p in pats:
            pattern_map[p].add(t)
    
    compat = defaultdict(list)
    for t1 in tokens:
        candidates = set()
        for p in token_patterns[t1]:
            candidates |= pattern_map[p]
        compat[t1] = [c for c in candidates if c != t1]
    return compat

def build_combos_graph(cross_combos):
    nodes, edges, locator = [], set(), {}
    nid = 0
    for token, combos in cross_combos.items():
        for idx, combo in enumerate(combos):
            locator[(token, idx)] = nid
            nodes.append({"id": nid, "token": token, "combo": combo})
            nid += 1

    compat = build_token_compatibility(cross_combos)
    seen = set()

    def shared_distance(tri, idx_pair):
        if idx_pair == (0,1):
            return tri[9]
        elif idx_pair == (1,2):
            return tri[11]
        else:
            raise AssertionError(f"Unexpected shared indices {idx_pair}, center must be shared")

    for t1, comp_list in compat.items():
        for t2 in comp_list:
            if (t2, t1) in seen:
                continue
            seen.add((t1, t2))

            c1_list, c2_list = cross_combos[t1], cross_combos[t2]
            iterator = (
                combinations(range(len(c1_list)), 2)
                if t1 == t2 else product(range(len(c1_list)), range(len(c2_list)))
            )

            for i, j in iterator:
                combo1, combo2 = c1_list[i], c2_list[j]
                first_idx_pair = None
                valid = True

                for tri1, tri2 in zip(combo1, combo2):
                    shared = set(tri1[:3]) & set(tri2[:3])
                    if len(shared) != 2 or tri1[1] not in shared or tri2[1] not in shared:
                        valid = False
                        break

                    idx1 = tuple(sorted(k for k,v in enumerate(tri1[:3]) if v in shared))
                    idx2 = tuple(sorted(k for k,v in enumerate(tri2[:3]) if v in shared))

                    if first_idx_pair is None:
                        first_idx_pair = (idx1, idx2)

                    elif (idx1, idx2) != first_idx_pair:
                        valid = False
                        break

                    d1 = shared_distance(tri1, idx1)
                    d2 = shared_distance(tri2, idx2)
                    if d1 != d2:
                        valid = False
                        break

                if not valid:
                    continue
               
                u1, c1, w1 = [], [], []
                u2, c2, w2 = [], [], []
                for tr1, tr2 in zip(combo1, combo2):
                    u1.append(tr1[0])
                    c1.append(tr1[1])
                    w1.append(tr1[2])

                    u2.append(tr2[0])
                    c2.append(tr2[1])
                    w2.append(tr2[2])
                
                u1, c1, w1 = tuple(u1), tuple(c1), tuple(w1)
                u2, c2, w2 = tuple(u2), tuple(c2), tuple(w2)
                for a, b in ((c1, u1), (c1, w1), (c2, u2), (c2, w2)):
                    if a == b:
                        continue
                    edge = tuple(sorted((a,b)))
                    edges.add(edge)

    # return {"nodes": nodes, "edges": edges}
    return edges

def create_std_matrix(nodes, matrices: dict, maps: dict, threshold: float = 3.0):
    dim = len(nodes[0])
    K = len(nodes) 

    maps_out = {}
    maps_out["possible_nodes"] = {}
    for i, node in enumerate(nodes):
        maps_out["possible_nodes"][i]      = node
        maps_out["possible_nodes"][str(node)] = i
    maps["possible_nodes"] = maps_out["possible_nodes"]
    
    stacked_pruned  = np.empty((dim, K, K))
    stacked_thresh  = np.empty_like(stacked_pruned)

    for p in range(dim):
        idx = [node[p] for node in nodes]

        for i in range(K):
            stacked_pruned[p,  i, :] = matrices["dm_pruned"][idx[i],  idx]
            stacked_thresh[p, i, :]  = matrices["dm_thresholded"][idx[i], idx]

    var_pruned = np.std(stacked_pruned, axis=0)    
    var_thresh = np.std(stacked_thresh, axis=0)  

    mask_pruned = np.any((stacked_pruned == 0) | np.isnan(stacked_pruned), axis=0)
    mask_thresh = np.any((stacked_thresh == 0) | np.isnan(stacked_thresh), axis=0)

    var_pruned = np.where(mask_pruned, np.nan, var_pruned)
    var_thresh = np.where(mask_thresh, np.nan, var_thresh)        

    mask_valid = (0 < var_pruned) & (var_pruned < threshold)
    mask_invalid = ~mask_valid
    var_pruned[mask_valid] = 1
    var_pruned[mask_invalid] = np.nan

    mask_valid = (0 < var_thresh) & (var_thresh < threshold)
    mask_invalid = ~mask_valid
    var_thresh[mask_valid] = 1
    var_thresh[mask_invalid] = np.nan
 
    new_matrices = {
        "dm_possible_nodes": var_pruned,
        "adj_possible_nodes": var_thresh
    }

    return new_matrices, maps

def association_product(graph_data: list,
                        config: dict,
                        debug: bool = True) -> Union[Dict[str, List], None]:
    logger = logging.getLogger("association.association_product")
    
    checks = config.get("checks", {"rsa": True, "depth": True})

    # residues_classes = create_residues_classes('resources/atchley_aa.csv', config["residues_similarity_cutoff"])

    if config["classes_path"] is not None:
        with open(config["classes_path"], "r") as f:
            classes = json.load(f)
    else:
        classes = {}


    depths = []
    if checks["depth"]:
        for gd in graph_data:
            df = gd["residue_depth"]
            depth_dict = dict(zip(df["ResNumberChain"], df["ResidueDepth"]))
            depths.append(
                np.array([ depth_dict[node] for node in gd["depth_nodes"] ])
            )

    graph_collection = {
        "graphs": [gd["graph"] for gd in graph_data],
        "triads": [find_triads(gd, classes, config, checks) for gd in graph_data],
        "contact_maps": [gd["contact_map"] for gd in graph_data],
        "residue_maps_all": [gd["residue_map_all"] for gd in graph_data],
        "rsa_maps": [gd["rsa"] for gd in graph_data],
        "nodes_graphs": [sorted(list(gd["graph"].nodes())) for gd in graph_data]
    }

    graph_collection["depths_maps"] = depths
    
    save("association_product", f"graph_collection", graph_collection)
    ranges_graph = indices_graphs(graph_collection["graphs"])
    total_length = sum(len(g.nodes()) for g in graph_collection["graphs"])
    metadata = {
        "total_length": total_length,
        "ranges_graph": ranges_graph
    }
  
    matrices_dict = {
        "type": 0,
        "neighbors": None,
        "rsa": None,
        "identity": None,
        "depth": None,
        "associated": None,
        "similarity": None,
        "dm_thresholded": None,
        "dm_pruned": None,
        "metadata": metadata
    }
    
    filter_input = {
        "contact_maps": graph_collection["contact_maps"],
        "rsa_maps": graph_collection["rsa_maps"],
        "residue_maps": graph_collection["residue_maps_all"],
        "depths_maps": graph_collection["depths_maps"],
        "nodes_graphs": graph_collection["nodes_graphs"]
    }
    
    logger.info("Creating pruned and thresholded arrays...")
    matrices_dict, maps = filter_maps_by_nodes(filter_input,
                                            distance_threshold=config["centroid_threshold"],
                                            matrices_dict=matrices_dict)
    logger.info("Arrays created successfully!")

    current_value = 0
    maps["residue_maps_unique_break"] = {} 
    for i, res_map in enumerate(maps["full_residue_maps"]):
        maps["residue_maps_unique"].update({val + current_value: key for key, val in res_map.items()})
        maps["residue_maps_unique_break"][i] = {val + current_value: key for key, val in res_map.items()}
        current_value += len(res_map)

    inv_maps = {
        k: { res: idx for idx, res in br.items() }
        for k, br in maps["residue_maps_unique_break"].items()
    }

    current_index = 0

    dm_thresh = np.zeros((metadata["total_length"], metadata["total_length"]))
    dm_prune = np.zeros((metadata["total_length"], metadata["total_length"]))

    save("association_product", f"maps", maps)
    save("association_product", f"inv_maps", inv_maps)

    for i, graph in enumerate(graph_collection["graphs"]):

        graph_length = len(graph.nodes())
        new_index = current_index + graph_length
        dm_thresh[current_index:new_index, current_index:new_index] = matrices_dict["thresholded_contact_maps"][i]
        dm_prune[current_index:new_index, current_index:new_index] = matrices_dict["pruned_contact_maps"][i]
        current_index = new_index

    save("association_product", f"dm_thresh", dm_thresh)
    save("association_product", f"dm_prune", dm_prune)
    matrices_dict["dm_thresholded"] = dm_thresh
    matrices_dict["dm_pruned"] = dm_prune

    cross_combos = cross_protein_triads(graph_collection["triads"])
    triad_graph = build_combos_graph(cross_combos)

    save("association_product", f"cross_combos", cross_combos)
    save("association_product", f"triad_graph", triad_graph)
    
    tuple_edges = [tuple(edge) for edge in triad_graph]
    log.debug(f"Number of edges: {len(tuple_edges)}")
    G = nx.Graph()
    G.add_edges_from(tuple_edges)
    components = list(nx.connected_components(G))
    save("comp_id_0", f"graph_associated_base", G)
    Graphs = [([G], 0)]
    comp_id = 1
    maps["inv_maps"] = inv_maps
    for component in components:
        log.debug(f"Processing component {comp_id} with {len(component)} nodes")

        subG = nx.Graph()
        subG.add_nodes_from(component)

        for u in component:
            for v in G.neighbors(u):
                if v in component:
                    subG.add_edge(u, v)

        dm_thresh_graph = np.zeros((metadata["total_length"], metadata["total_length"]))
        for u, v in subG.edges():
            for p, (res_u, res_v) in enumerate(zip(u, v)):
                if res_u != res_v:
                    split_res_u, split_res_v = res_u.split(":"), res_v.split(":")
                    res_u_tuple = (split_res_u[0], split_res_u[2], split_res_u[1])
                    res_v_tuple = (split_res_v[0], split_res_v[2], split_res_v[1])
                    idx_u = inv_maps[p][res_u_tuple]
                    idx_v = inv_maps[p][res_v_tuple]
                    dm_thresh_graph[idx_u, idx_v] = dm_thresh[idx_u, idx_v]
                    dm_thresh_graph[idx_v, idx_u] = dm_thresh[idx_v, idx_u]

        save(f"comp_id_{comp_id}", f"dm_thresh_graph", dm_thresh)
        matrices_dict["dm_thresholded"] = dm_thresh_graph
    
        nodes = list(subG.nodes())
        nodes_indices = []

        for node in nodes:
            node_converted = []
            for k, res in enumerate(node):
                res_split = res.split(":")
                res_tuple = (res_split[0], res_split[2], res_split[1])
                res_indice = inv_maps[k][res_tuple]
                node_converted.append(res_indice)
            nodes_indices.append(node_converted)

        save(f"comp_id_{comp_id}", f"nodes_indices", nodes_indices)

        matrices_mul, maps_mul = create_std_matrix(
            nodes=nodes_indices,
            matrices=matrices_dict,
            maps=maps,
            threshold=config["distance_diff_threshold"]
        )

        save(f"comp_id_{comp_id}", f"matrices_mul", matrices_mul)
        save(f"comp_id_{comp_id}", f"maps_mul", maps_mul)

        frames = generate_frames(
            matrices=matrices_mul,
            maps=maps_mul,
        )

        if len(frames.keys()) > 1:
            Graphs.extend([(create_graph(frames, typeEdge="edges_residues", comp_id=comp_id), comp_id)])
            comp_id += 1

    save("association_product", f"Graphs", Graphs)
    return {
            "AssociatedGraph": Graphs
        }

def generate_frames(matrices, maps):
    """
    Extrai frames de arestas a partir das matrizes filtradas,
    usando convert_edges_to_residues para montar edges_indices e edges_residues.
    """
    dm  = matrices["dm_possible_nodes"].copy()
    adj = matrices["adj_possible_nodes"].copy()
    np.fill_diagonal(dm,  1)
    np.fill_diagonal(adj, np.nan)
    K = dm.shape[0]

    frames = {}

    edges_base = set()
    for i in range(K):
        js = np.where(adj[i] == 1)[0]
        for j in js:
            edges_base.add(frozenset((i, j)))

    edges_original_0, edges_idx_0, edges_res_0 = convert_edges_to_residues(edges_base, maps)
    frames[0] = {
        "edges_indices": edges_idx_0,
        "edges_residues": edges_res_0,
    }

    seen_edge_sets = set()
    checked_node_sets = set()
    k = 1

    for i in range(K):
        inter     = dm[i].copy()

        accepted  = {i}
        frontier  = set(np.where(adj[i] == 1)[0])

        if not frontier:
            continue
        visited = set()

        while frontier:
            frontier -= visited
            if not frontier:
                break
            for u in frontier:
                inter *= dm[u] 

            visited |= frontier
            accepted |= frontier

            next_layer = set()
            for u in frontier:
                nbrs = set(np.where(adj[u] == 1)[0])
                next_layer |= nbrs
            frontier = next_layer & set(np.where(inter == 1)[0])

        inter_idx = np.where(inter == 1)[0]

        if inter_idx.size < 4:
            continue

        node_set = frozenset(inter_idx)
        if node_set in checked_node_sets:
            continue
        checked_node_sets.add(node_set)

        nodes_idx = np.array(sorted(inter_idx), dtype=int)
        sub_adj   = adj[np.ix_(nodes_idx, nodes_idx)]
        degrees   = np.sum(np.nan_to_num(sub_adj), axis=1)
        valid_mask = (degrees > 1)
        
        valid_nodes = nodes_idx[valid_mask]
        filtered    = adj[np.ix_(valid_nodes, valid_nodes)]
        
        edges = {
            frozenset((int(valid_nodes[p]), int(valid_nodes[q])))
            for p, q in zip(*np.where(filtered == 1))
        }
        frame_edges = {e for e in edges}
        if len(frame_edges) < 4:
            continue

        edge_key = frozenset(frame_edges)
        if edge_key in seen_edge_sets:
            continue
        seen_edge_sets.add(edge_key)

        _, edges_idx, edges_res = convert_edges_to_residues(frame_edges, maps, True)
        frames[k] = {
            "edges_indices": edges_idx,
            "edges_residues": edges_res,
        }
        k += 1

    others = sorted(
        (fid for fid in frames if fid != 0),
        key=lambda fid: len(frames[fid]["edges_indices"]),
        reverse=True
    )
    ordered = {0: {"edges_indices": [], "edges_residues": []}}
    for new_id, old_id in enumerate(others, start=1):
        ordered[new_id] = frames[old_id]
    
    final_frames = {}
    seen = []   
    i = 0

    for k, frame in ordered.items():
        edges_set = frozenset(map(tuple, frame["edges_indices"]))

        if any(edges_set.issubset(kept) for kept in seen):
            continue

        to_remove = [kept for kept in seen if kept.issubset(edges_set) and kept != edges_set]
        for small in to_remove:
            seen.remove(small)
            for idx, f in list(final_frames.items()):
                if frozenset(map(tuple, f["edges_indices"])) == small:
                    logging.debug(f"Removing {final_frames[idx]} from {small}")
                    del final_frames[idx]

        seen.append(edges_set)
        final_frames[i] = frame
        i += 1
    
    final_frames[0] = frames[0]
    
    return final_frames

def create_graph(edges_dict: Dict, typeEdge: str = "edges_indices", comp_id = 0):
    Graphs = []
    k = 0
    for frame in range(0, len(edges_dict.keys())):
        edges = edges_dict[frame][typeEdge]    
        
        G_sub = nx.Graph()  
        
        if len(edges) > 1:
                
            for sublist in edges:
                sublist = list(sublist)

                node_a = tuple(sublist[0]) if isinstance(sublist[0], np.ndarray) else sublist[0]
                node_b = tuple(sublist[1]) if isinstance(sublist[1], np.ndarray) else sublist[1] 
                G_sub.add_edge(node_a, node_b)
                
            chain_color_map = {}
            color_palette = plt.cm.get_cmap('tab10', 20) 
            color_counter = 1 
            
            if typeEdge == "edges_residues":
                for nodes in G_sub.nodes:
                    chain_id = nodes[0][0]+nodes[1][0]
                    
                    if chain_id not in chain_color_map and chain_id[::-1] not in chain_color_map:
                        chain_color_map[chain_id] = color_palette(color_counter)[:3]
                        chain_color_map[chain_id[::-1]] = chain_color_map[chain_id]  # RGB tuple
                        # log.debug(f"Chain_id: {chain_id}, color: {chain_color_map[chain_id]}")
                        color_counter += 1

                    G_sub.nodes[nodes]['chain_id'] = chain_color_map[chain_id]
            
            G_sub.remove_nodes_from(list(nx.isolates(G_sub)))
            log.debug(f"{comp_id} Number of nodes graph {k}: {len(G_sub.nodes)}")
            k+= 1

            if k >= 100:
                break
            Graphs.append(G_sub)
    return Graphs

def build_contact_map(
    pdb_file: str,
    *,
    exclude_waters: bool = True,
    atom_preference: Tuple[str, str] = ("CB", "CA"),
    water_atom_preference: Tuple[str, ...] = ("O", "OW", "OH2"),
    fallback_any_atom: bool = True,
) -> Tuple[np.ndarray, Dict[Tuple[str, int], int], Dict[Tuple[str, int, str], int]]:
    """
    Build a residue–residue distance (contact) map using representative atoms.

    Parameters
    ----------
    pdb_file : str
        Path to a PDB file.
    include_waters : bool, default=False
        If True, water residues are included using an oxygen atom as representative.
    atom_preference : (str, str), default=("CB", "CA")
        Ordered preference of atom names for standard residues.
    water_atom_preference : tuple of str, default=("O", "OW", "OH2")
        Ordered preference of atom names for water residues.
    fallback_any_atom : bool, default=True
        If no preferred atoms are present, fall back to the first atom available.

    Returns
    -------
    contact_map : ndarray of shape (N, N)
        Symmetric matrix of pairwise Euclidean distances (Å) between representative atoms.
    residue_map_dict : dict
        Mapping ``(chain_id, residue_number) -> index``.
    residue_map_dict_all : dict
        Mapping ``(chain_id, residue_number, residue_name) -> index``.

    Notes
    -----
    - Waters are identified by `hetfield == 'W'` or residue name in
      {'HOH', 'H2O', 'WAT', 'TIP3', 'SOL'}.
    - Only the first model is used for deterministic behavior.
    - Residues missing all representative options are skipped.
    """
    from Bio.PDB import PDBParser

    WATER_NAMES = {"HOH", "H2O", "WAT", "TIP3", "SOL"}

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # Use only the first model
    model = next(iter(structure))

    entries: List[Tuple[str, str, str, np.ndarray]] = []

    for chain in model:
        chain_id = chain.id
        for residue in chain:
            hetfield, resseq, _icode = residue.id  # (het, number, icode)
            res_name = residue.get_resname().strip()
            is_water = (hetfield == "W") or (res_name in WATER_NAMES)

            # Skip waters if not requested
            if is_water and exclude_waters:
                continue

            coord = None

            if is_water:
                # Prefer oxygen-like atom names for waters
                for atom_name in water_atom_preference:
                    if residue.has_id(atom_name):
                        coord = residue[atom_name].get_coord()
                        break
            else:
                # Standard residues: CB then CA
                for atom_name in atom_preference:
                    if residue.has_id(atom_name):
                        coord = residue[atom_name].get_coord()
                        break

            # Fallback: any atom available (useful for ligands or odd residues)
            if coord is None and fallback_any_atom:
                try:
                    atom = next(residue.get_atoms())
                    coord = atom.get_coord()
                except StopIteration:
                    pass

            if coord is None:
                # No representative atom found; skip this residue
                continue

            residue_full = residue.get_full_id()
            icode = residue_full[-1][-1] 
            res_id = f"{resseq}{icode.strip()}" if icode.strip() else str(resseq)
            entries.append((chain_id, res_id, res_name, np.asarray(coord, dtype=float)))


    # Deterministic ordering: by chain, residue number, residue name
    entries.sort(key=lambda x: (x[0], int(''.join(filter(str.isdigit, x[1]))), ''.join(filter(str.isalpha, x[1])), x[2]))

    if not entries:
        return np.zeros((0, 0), dtype=float), {}, {}

    coords = np.vstack([e[3] for e in entries])  # (N, 3)
    residue_map = [(e[0], e[1]) for e in entries]
    residue_map_all = [(e[0], e[1], e[2]) for e in entries]

    residue_map_dict: Dict[Tuple[str, str], int] = {t: i for i, t in enumerate(residue_map)}
    residue_map_dict_all: Dict[Tuple[str, str, str], int] = {t: i for i, t in enumerate(residue_map_all)}

    # Vectorized pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    contact_map = np.sqrt(np.sum(diff * diff, axis=2, dtype=float), dtype=float)

    return contact_map, residue_map_dict, residue_map_dict_all

def get_residues_within_distance(pdb_file, chain_ids, target_chain_ids, distance_threshold):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', pdb_file)

    chains = {chain.id: chain for chain in structure.get_chains() if chain.id in chain_ids}
    target_chains = {chain.id: chain for chain in structure.get_chains() if chain.id in target_chain_ids}

    result = set()

    for chain_id, chain in chains.items():
        for residue in chain:
            for target_chain_id, target_chain in target_chains.items():
                for target_residue in target_chain:
                    if residue != target_residue:
                        for atom in residue:
                            for target_atom in target_residue:
                                distance = atom - target_atom
                                if distance < distance_threshold:
                                    result.add(f"{chain_id}:{residue.resname}:{residue.id[1]}")
                                    break

    return list(result)

def get_side_chain_atoms(residue):
    if residue.resname == "GLY":
        return list(residue)
    side_chain_atoms = []
    for atom in residue:
        atom_name = atom.get_name()
        if atom_name != "CA" and atom_name != "C" and atom_name != "N" and atom_name != "O":
            side_chain_atoms.append(atom)
    return side_chain_atoms

def get_residues_within_distance_sidechain(pdb_file, chain_ids, target_chain_ids, distance_threshold):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', pdb_file)

    chains = {chain.id: chain for chain in structure.get_chains() if chain.id in chain_ids}
    target_chains = {chain.id: chain for chain in structure.get_chains() if chain.id in target_chain_ids}

    result = set()

    for chain_id, chain in chains.items():
        for residue in chain:
            side_chain_atoms = get_side_chain_atoms(residue)
            for target_chain_id, target_chain in target_chains.items():
                for target_residue in target_chain:
                    if residue != target_residue:
                        target_side_chain_atoms = get_side_chain_atoms(target_residue)
                        for atom in side_chain_atoms:
                            for target_atom in target_side_chain_atoms:
                                distance = atom - target_atom
                                if distance < distance_threshold:
                                    result.add(f"{chain_id}:{residue.resname}:{residue.id[1]}")
                                    break

    return list(result)

def get_residues_within_distance_CB(pdb_file, chain_ids, target_chain_ids, distance_threshold):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', pdb_file)

    chains = {chain.id: chain for chain in structure.get_chains() if chain.id in chain_ids}
    target_chains = {chain.id: chain for chain in structure.get_chains() if chain.id in target_chain_ids}

    result = set()

    for chain_id, chain in chains.items():
        for residue in chain:
            for target_chain_id, target_chain in target_chains.items():
                for target_residue in target_chain:
                    if residue != target_residue:
                        # Get the atom to calculate distance from for each residue type
                        if residue.resname == "GLY":
                            residue_atom = residue["CA"]
                        else:
                            residue_atom = residue["CB"]

                        if target_residue.resname == "GLY":
                            target_residue_atom = target_residue["CA"]
                        else:
                            target_residue_atom = target_residue["CB"]

                        distance = residue_atom - target_residue_atom
                        if distance < distance_threshold:
                            result.add(f"{chain_id}:{residue.resname}:{residue.id[1]}")
                            break

    return list(result)

def create_sphere_residue(residue_name, residue_number, coord):
    atom = Atom("CA", coord, 1.0, 1.0, " ", "DUM", 1, element='CA')
    residue = Residue((' ', int(residue_number), ' '), residue_name, " ")
    residue.add(atom)
    return residue

def align_structures_by_chain(reference_pdb, target_pdb, chain_id):
    """
    Aligns a target structure to a reference structure based on a specified chain ID.

    Args:
        reference_pdb (str): Path to the reference PDB file.
        target_pdb (str): Path to the target PDB file.
        chain_id (str): The chain ID to use for alignment.

    Returns:
        Superimposer: The Superimposer object after alignment.
    """
    # Create a PDB parser
    parser = PDBParser(QUIET=True)

    # Load the structures
    reference_structure = parser.get_structure('reference', reference_pdb)
    target_structure = parser.get_structure('target', target_pdb)

    # Extract the chains
    reference_chain = reference_structure[0][chain_id]
    target_chain = target_structure[0][chain_id]

    # Extract the CA atoms from both chains for alignment
    reference_atoms = [atom for atom in reference_chain.get_atoms() if is_aa(atom.get_parent()) and atom.get_id() == 'CA']
    target_atoms = [atom for atom in target_chain.get_atoms() if is_aa(atom.get_parent()) and atom.get_id() == 'CA']

    if len(reference_atoms) != len(target_atoms):
        raise ValueError("The number of CA atoms in the chains do not match. Ensure both chains have the same number of residues.")

    # Create a Superimposer object
    super_imposer = Superimposer()

    # Set the reference and target atoms for alignment
    super_imposer.set_atoms(reference_atoms, target_atoms)

    # Apply the transformation to the target structure
    super_imposer.apply(target_structure.get_atoms())

    # Print the RMSD
    print(f"RMSD: {super_imposer.rms:.4f} Å")

    return super_imposer

def add_sphere_residues(graphs, list_node_names_mol, output_path, node_name):
    
    for graph, node_names_mol in zip(graphs, list_node_names_mol):
        mol_path = graph[1]
        
        # Read PDB file
        parser = PDBParser()
        structure = parser.get_structure('protein', mol_path)

        # Create a new structure to hold the spheres
        new_structure = Structure.Structure("spheres")
        
        # Create a new model and chain for each mol
        new_model = Model.Model(0)
        new_chain = Chain.Chain('X')
        new_model.add(new_chain)
        new_structure.add(new_model)

        # Keep track of added residues to avoid duplicates
        added_residues = set()

        # Add sphere residues to the new structure
        for residue_info in node_names_mol:
            chain_id, residue_name, residue_number = residue_info.split(':')
            residue_key = (chain_id, int(residue_number))
            if residue_key not in added_residues:
                residue = structure[0][chain_id][int(residue_number)]
                #ca_atom = residue['CA']
                atom_coords = [atom.coord for atom in residue]
                centroid_coords = np.mean(atom_coords, axis=0)
                sphere_residue = create_sphere_residue(residue_name, residue_number, centroid_coords)
                new_chain.add(sphere_residue)
                added_residues.add(residue_key)

        name = mol_path.replace("\\", "_").replace("/", "_")
        # Write the new PDB file
        io = PDBIO()
        io.set_structure(new_structure)
        io.save(path.join(output_path,f'spheres_{name}_{node_name}.pdb'))
    
def create_subgraph_with_neighbors(graphs, association_graph, node, max_nodes):
    """
    This is mostly an implementation of the BFS algorithm with some modification
    
    Parameters:
        graph (nx.Graph): The original graph.
        node: The node for which the subgraph is to be created.
        max_nodes (int): The maximum number of nodes allowed in the subgraph.
        
    Returns:
        nx.Graph: 
    """
    # Initialize a set to store visited nodes
    visited = set()
    
    # Initialize the subgraph with the given node
    subgraph = nx.Graph()
    subgraph.add_node(node)
    visited.add(node)
    
    # Queue to store nodes to visit next
    queue = [(node, None)]  # (node, parent)


    while queue and subgraph.number_of_nodes() < max_nodes: #the max nodes only act here when start the new layer (getting the new current node)
        # Once it goes inside the neighbor loop, it will finish it even that it means to generate more neigbors than expected
        # Pop the node and its parent from the queue
        current_node, parent = queue.pop(0)
        
        # If not the starting node, add edge to its parent
        if parent is not None:
            subgraph.add_edge(current_node, parent)
        
        # Iterate over neighbors of the current node
        neighbors = list(association_graph.neighbors(current_node))

        #get euclidian distance between current node and neighbors to sort the neighbors list 
        #Mol A
        dists = []
        
        for graph in graphs:
            graph_matrix = graph[0].graph["pdb_df"]
            current_node_index = graph_matrix[graph_matrix['node_id'] == current_node[0]].index[0]
            neighbor_indices = [graph_matrix[graph_matrix['node_id'] == nodes[0]].index[0] for nodes in neighbors if nodes != current_node]
            dist = compute_distmat(graph_matrix).iloc[neighbor_indices, current_node_index]
            dists.append(dist)

        # Get a average distance for sorting
        average_list = sum(dists)/len(graphs)
        
        # Pair each nodes with its corresponding numeric value
        paired_list = list(zip(average_list, neighbors))

        # Sort the pairs based on the numeric values
        paired_list.sort()

        # Extract the sorted list of nodes
        neighbors_sorted = [string for _, string in paired_list]

        for neighbor in neighbors_sorted:
            # If neighbor is not visited and adding it won't exceed max_nodes
            # if neighbor not in visited and subgraph.number_of_nodes() + 1 <= max_nodes:
            if neighbor not in visited: #try to visit all in the neighbor layer
                # Add the neighbor to the subgraph
                subgraph.add_node(neighbor)
                
                # Mark neighbor as visited
                visited.add(neighbor)
                
                # Add neighbor to the queue -> so that the neighbor will become current nodes and generate new neighbors
                queue.append((neighbor, current_node))
    
    # Add edges between selected nodes based on the original graph
    for u in subgraph.nodes():
        for v in subgraph.nodes():
            if u != v and association_graph.has_edge(u, v):
                subgraph.add_edge(u, v)
    
    return subgraph

def convert_1aa3aa(AA):
    amino_acid_codes = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'E': 'GLU',
    'Q': 'GLN',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL'}
    
    return amino_acid_codes[AA]

def convert_3aa1aa(AA):
    amino_acid_codes = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLU': 'E',
    'GLN': 'Q',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V'}

    return amino_acid_codes[AA]

def normalize_rows(arr):
    # Initialize an array to hold the normalized values
    normalized_array = np.zeros_like(arr, dtype=float)
    
    # Iterate over each row
    for i in range(arr.shape[0]):
        row = arr[i, :]
        min_val = np.min(row)
        max_val = np.max(row)
        
        # Apply the normalization formula to each row
        normalized_array[i, :] = (row - min_val) / (max_val - min_val)
        
    return normalized_array

def calculate_atchley_average(node, read_emb):
    # Inicializar um vetor para somar os fatores de Atchley
    atchley_factors_sum = np.zeros(read_emb.shape[1] - 1)  # Exclui a coluna de aminoácidos
    
    if isinstance(node, tuple):
        # Iterar sobre cada átomo no nó (tupla)
        for atom in node:
            residue = convert_3aa1aa(atom.split(":")[1])
            
            # Obter os fatores de Atchley para o resíduo
            atchley_factors = read_emb[read_emb.AA == residue].iloc[:, 1:].values[0]
            
            # Somar os fatores de Atchley
            atchley_factors_sum += atchley_factors
        
        # Calcular a média dos fatores de Atchley
        atchley_factors_avg = atchley_factors_sum / len(node)
    else:
        residue = convert_3aa1aa(node.split(":")[1])
        atchley_factors_avg = read_emb[read_emb.AA == residue].iloc[:, 1:].values[0]
    
    return np.asarray(atchley_factors_avg, dtype=np.float64)

def graph_message_passing(graph, embedding_path, use_degree, norm_features):
    '''
    This function performs a single message passing in the graph nodes to update node features.
    It returns a dictionary with the updated node features.
    '''
    # Obtém a matriz de adjacência
    order = sorted(list(graph.nodes()))
    adj = nx.adjacency_matrix(graph, nodelist=order).todense().astype(np.float64)
    
    # Obtém a matriz de distâncias
    pdb_df = graph.graph["pdb_df"]
    pdb_df = pdb_df.set_index('node_id', inplace=False)
    
    ordered_pdb_df = pdb_df.reindex(order)
    dist_df = compute_distmat(ordered_pdb_df)
    dist_m = np.array(dist_df.values.tolist(), dtype=np.float64)
    
    # Define um epsilon para evitar divisão por zero
    epsilon = 1e-8
    
    # Multiplicação elemento a elemento (Hadamard) e divisão, somando epsilon no denominador
    mult = 1 / (np.array(adj) * dist_m + epsilon)
    mult[mult == np.inf] = 0  # Substitui infinito por 0
    
    # Normaliza as linhas para obter pesos entre 0 e 1
    row_sums = np.sum(mult, axis=1)
    weights_m = mult / row_sums[:, np.newaxis]
    
    # Lê a embedding e constrói a matriz de features
    read_emb = pd.read_csv(embedding_path)
    feature_matrix = np.array([
        read_emb[read_emb.AA == convert_3aa1aa(node.split(":")[1])].iloc[:,1:].values.tolist()[0]
        for node in order
    ])
        
    # Message passing: multiplica a matriz de pesos pela matriz de features
    message_passing_m = weights_m @ feature_matrix
    
    # Concatena as features originais com as atualizadas
    concat_feat_matrix = np.concatenate((feature_matrix, message_passing_m), axis=1)
    
    # Converte a matriz em dicionário usando os nomes dos nós ordenados
    node_names = list(order)
    assert len(node_names) == concat_feat_matrix.shape[0], "Number of keys must match the number of rows in the array."
    
    if use_degree and norm_features:
        # Adiciona a contagem de vizinhos normalizada como feature
        neighbors_count = {node: (len(list(graph.neighbors(node))) - 1) / 10 for node in sorted(list(graph.nodes()))}
        concat_feat_matrix_norm = normalize_rows(concat_feat_matrix)
        feat_MP_dict = {
            node_names[i]: np.concatenate([concat_feat_matrix_norm[i, :], [neighbors_count[node_names[i]]]])
            for i in range(concat_feat_matrix_norm.shape[0])
        }
    else:
        feat_MP_dict = {
            node_names[i]: concat_feat_matrix[i, :]
            for i in range(concat_feat_matrix.shape[0])
        }
    
    return feat_MP_dict

def cosine_similarity2(array1, array2):
    # Ensure the arrays are 1-dimensional
    assert array1.ndim == 1 and array2.ndim == 1, "Arrays must be 1-dimensional"
    
    # Compute the dot product
    dot_product = np.dot(array1, array2)
    
    # Compute the L2 norms (Euclidean norms)
    norm1 = np.linalg.norm(array1)
    norm2 = np.linalg.norm(array2)
    
    # Compute the cosine similarity
    similarity = dot_product / (norm1 * norm2)
    
    return similarity

def check_similarity(node_pair, rep_molA, rep_molB, threshold):
    '''
    Takes a tuple of nodes in the following format: 'A:PRO:57', 'A:THR:178')
    '''
    # print(f"node_pair: {node_pair}\nrep_molA{rep_molA}")
    pairA = "|".join(node_pair[0]) if isinstance(node_pair[0], tuple) else node_pair[0] 
    pairB = "|".join(node_pair[1]) if isinstance(node_pair[1], tuple) else node_pair[1] 
    cos_sim = cosine_similarity(rep_molA[pairA], rep_molB[pairB])
    #debug
    if cos_sim > threshold:
        print(pairA)
        print(pairB)
        print(rep_molA[pairA])
        print(rep_molB[pairB])
        print(cos_sim)

    if cos_sim > threshold:
        return True
    else:
        return False

def get_coords_xyz(node_ID, graph):
    """
    This function gets the ID of a node and a Graphein graph object and returns the xyz coordinates of that node
    """
    pdb_df = graph.graph["pdb_df"]  # Obtém o DataFrame de coordenadas

    # Filtra as linhas onde 'node_id' corresponde ao node_ID
    node_data = pdb_df[pdb_df['node_id'] == node_ID]

    if node_data.empty:
        log.error(f"Node ID {node_ID} não encontrado no pdb_df!")

    # Obtém as coordenadas
    x_coord = node_data['x_coord'].iloc[0]
    y_coord = node_data['y_coord'].iloc[0]
    z_coord = node_data['z_coord'].iloc[0]

    return np.array([x_coord, y_coord, z_coord])

def angle_between_vectors(a, b):
    """
    This function get two vectors and returns the angle betweem them
        """
    # Normalize the vectors
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    # Ensure norms are not zero to avoid division by zero
    if a_norm == 0 or b_norm == 0:
        raise ValueError("Vectors must not be zero vectors")

    # Compute dot product and angle
    dot_product = np.dot(a, b)
    cos_angle = dot_product / (a_norm * b_norm)

    # Clip cos_angle to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Return the angle degrees
    return np.degrees(np.arccos(cos_angle))

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def filter_nodes_angle(G: nx.Graph, graphs: List[nx.Graph], angle_diff: float):
    ang_node_dict = {}  # Dicionário para armazenar as diferenças de ângulos

    log.debug("Filtering angles")
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            continue  # Precisamos de pelo menos 2 vizinhos para calcular ângulos

        ang_diffs = []  # Lista para armazenar as diferenças de ângulo

        # Percorre todos os pares de vizinhos do nó
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                angles = []  # Lista para armazenar os ângulos de cada grafo

                # Calcula os ângulos para todos os grafos em 'graphs'
                for k, graph in enumerate(graphs):
                    n1, n2 = neighbors[i][k], neighbors[j][k]  # Pega os nós correspondentes no grafo k
                    
                    # Obtém as coordenadas do nó e dos vizinhos
                    coord_node = get_coords_xyz(node[k], graph)
                    coord_n1 = get_coords_xyz(n1, graph)
                    coord_n2 = get_coords_xyz(n2, graph)

                    # Calcula os vetores
                    v1 = coord_n1 - coord_node
                    v2 = coord_n2 - coord_node

                    angle = angle_between_vectors(v1, v2)
                    angles.append(angle)

                for angle1, angle2 in combinations(angles, 2):
                    ang_diffs.append(abs(angle1 - angle2))

        ang_node_dict[node] = ang_diffs

    filtered_nodes_ang = [
        key for key, values in ang_node_dict.items() if all(value < angle_diff for value in values)
    ]

    non_compliant_nodes = [
        key for key, values in ang_node_dict.items() if any(value >= angle_diff for value in values)
    ]

    non_compliant_nodes_sorted = sorted(non_compliant_nodes, key=lambda n: len(G[n]))

    # Cria um dicionário com os nós e seus respectivos vizinhos
    nodes_data = {str(node): list(G.neighbors(node)) for node in non_compliant_nodes_sorted}

    # Salva o dicionário como JSON
    with open("non_compliant_nodes.json", "w", encoding="utf-8") as f:
        json.dump(nodes_data, f, indent=4, ensure_ascii=False)

    return filtered_nodes_ang
