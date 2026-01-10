from core.tracking import save
from Bio.PDB import *
from collections import defaultdict
import numpy as np
from itertools import combinations, product
from Bio.PDB.vectors import Vector
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
import networkx as nx
import time
# from graphein.protein.edges.distance import compute_distmat
from os import path
import os
import pandas as pd
from collections import Counter
from typing import Any, FrozenSet, Tuple, List, Optional, Union, Dict, Set, Iterable, Sequence
import logging
import matplotlib.pyplot as plt
# from utils.cutils.combinations_filter import filtered_combinations
# from SERD_Addon.classes import StructureSERD
from sklearn.metrics.pairwise import cosine_similarity
import json
import math
from statistics import pstdev
import bisect
from utils.vis_tracer import TraversalTracer


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

def convert_edges_to_residues(edges: Set[FrozenSet], maps: Dict) -> Tuple[List, List, List]:
    """Convert the edges that contains tuple of indices to tuple of residues

    Args:
        edges (List[Tuple]): A list that contains tuple of edges that are made of tuples of indices
        maps (Dict): A map that relates the indice to residue

    Returns:
        convert_edge (List[Tuple]): Return edges converted to residues notation
    """
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
        # if set(converted_node1) != set(converted_node2) and check_cross_positions((converted_node1, converted_node2)):
        # if check_cross_positions((converted_node1, converted_node2)):
        original_edges.append(edge)
        edges_indices.append((converted_node1_indice, converted_node2_indice))
        converted_edges.append((converted_node1, converted_node2))
        # else:
            # log.debug(f'Invalid edge: {edge}, {converted_node1}:{converted_node2}')

    return original_edges, edges_indices, converted_edges
    
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


def value_to_class(
    value: float,
    bin_width: float,        # Fixed size of each division
    threshold: float,
    inverse: bool = False,
    upper_bound: float = 100.0,
    close_tolerance: float = 0.1,  # Absolute tolerance in 'value' units
) -> Optional[Union[int, List[int]]]:
    """
    Non-inverse:
      - Domain: (0, threshold]
      - Bins: (L, R] spaced by bin_width. Last bin may be smaller than bin_width.
      - Uncertainty: half of the bin_width.
      - If |value - center_of_bin| <= close_tolerance, returns [class_index].
      - Otherwise, returns all classes whose bins intersect with [value - inc, value + inc].

    Inverse:
      - Domain: [threshold, upper_bound]
      - Same multi-class behavior as non-inverse, but bins live on [threshold, upper_bound].
    """
    if bin_width <= 0:
        return None

    if not inverse:
        # Out of bounds check
        if value <= 0 or value > threshold:
            return None

        # Calculate total number of divisions
        n_divisions = math.ceil(threshold / bin_width)

        # Identify the primary bin index (1-based)
        i = math.ceil(value / bin_width)
        i = max(1, min(i, n_divisions))

        # Actual bin boundaries (last bin may be smaller)
        current_bin_left = (i - 1) * bin_width
        current_bin_right = min(i * bin_width, threshold)
        current_bin_actual_width = current_bin_right - current_bin_left

        bin_center = current_bin_left + (current_bin_actual_width / 2.0)

        # Clamp tolerance to at most half the actual bin width
        tol = max(0.0, float(close_tolerance))
        tol = min(tol, (current_bin_actual_width / 2.0) - 1e-12)

        if abs(value - bin_center) <= tol:
            return [i]

        inc = bin_width / 2.0
        low = value - inc
        high = value + inc

        classes: List[int] = []
        for j in range(1, n_divisions + 1):
            L = (j - 1) * bin_width
            R = min(j * bin_width, threshold)
            if (low < R) and (high > L):
                classes.append(j)

        return classes if classes else None

    # Inverse logic: mapping values from threshold up to upper_bound
    if value < threshold or value > upper_bound:
        return None

    span = upper_bound - threshold
    if span <= 0:
        return None

    n_divisions = math.ceil(span / bin_width)

    # Put the inverse domain on a local axis starting at 0
    rel_value = value - threshold

    # Primary bin index (1-based)
    i = math.ceil(rel_value / bin_width)
    i = max(1, min(i, n_divisions))

    # Actual bin boundaries in the inverse domain (last bin may be smaller)
    current_bin_left = (i - 1) * bin_width
    current_bin_right = min(i * bin_width, span)
    current_bin_actual_width = current_bin_right - current_bin_left

    bin_center = current_bin_left + (current_bin_actual_width / 2.0)

    tol = max(0.0, float(close_tolerance))
    tol = min(tol, (current_bin_actual_width / 2.0) - 1e-12)

    if abs(rel_value - bin_center) <= tol:
        return [i]

    inc = bin_width / 2.0
    low = rel_value - inc
    high = rel_value + inc

    # Clamp to inverse domain [0, span]
    low = max(0.0, low)
    high = min(span, high)

    classes: List[int] = []
    for j in range(1, n_divisions + 1):
        L = (j - 1) * bin_width
        R = min(j * bin_width, span)
        if (low < R) and (high > L):
            classes.append(j)

    return classes if classes else None
   
def dynamic_distance_to_classes(
    distance_classes: Dict[str, Dict[str, float]],
    value: float,
    diff_threshold: float,
    close_tolerance: float,
) -> Optional[Union[str, List[str]]]:
    """
    distance_classes: { class_name: {"low": float, "high": float}, ... }
    value: distance value (d1, d2, d3, ...)
    diff_threshold: distance_diff_threshold (max allowed distance diff between triads)
    close_tolerance: tolerance to say "this value is clearly in this bin"

    Returns:
        - None  -> no valid bin
        - "bin_k" or ["bin_k"]  -> single bin
        - ["bin_i", "bin_j", ...] -> multiple bins (near boundary)
    """
    # Filter only bins that could reasonably contain value
    candidate_bins = []
    for name, interval in distance_classes.items():
        low = interval["low"]
        high = interval["high"]
        if low <= value <= high:
            center = 0.5 * (low + high)
            candidate_bins.append((name, low, high, center))

    if not candidate_bins:
        # Value is outside all dynamic bins
        return None

    # Choose primary bin: the one whose center is closest to value
    primary = min(candidate_bins, key=lambda x: abs(value - x[3]))
    primary_name, primary_low, primary_high, primary_center = primary

    # If clearly inside this bin (center region), return only this one
    if abs(value - primary_center) <= close_tolerance:
        return [primary_name]

    # Otherwise, build an uncertainty interval based on diff_threshold
    low_unc = value - diff_threshold
    high_unc = value + diff_threshold

    # Collect all bins that intersect [low_unc, high_unc]
    hits: List[str] = []
    for name, interval in distance_classes.items():
        low = interval["low"]
        high = interval["high"]
        # intersection check between [low_unc, high_unc] and [low, high]
        if (low_unc <= high) and (high_unc >= low):
            hits.append(name)

    if not hits:
        return None

    return hits if len(hits) > 1 else hits[0]

def create_classes_bin(total_length: float, bin_width: float): 
    """Creates a dictionary of bins with their respective ranges."""
    n_bins = math.ceil(total_length / bin_width)
    return {
        str(n + 1): (n * bin_width, min((n + 1) * bin_width, total_length))
        for n in range(n_bins)
    }
 
# def find_class(dictionary, value):
#     for key in dictionary.keys():
#         range_tuple = dictionary[key]
#         if range_tuple[0] < value <= range_tuple[1]:
#             return key
#     else:
#         return None

def find_class(classes: Dict[str, Dict[str, float]], value: float):
    """
    Return all class names whose interval contains value.
    Assumes structure:
        classes = { "bin_1": {"low": ..., "high": ...}, ... }
    """
    hits = []
    for name, interval in classes.items():
        low = interval["low"]
        high = interval["high"]
        # choose your convention: [low, high] or [low, high)
        if low <= value <= high:
            hits.append(name)
    if not hits:
        return None
    # Return single class or list depending on your current semantics
    return hits[0] if len(hits) == 1 else hits


def residue_to_tuple(res):
    res_split = res.split(":")

    return (res_split[0], res_split[2], res_split[1])

def _as_list(x):
    """Converte None -> [], int -> [int], lista/tupla/conjunto -> lista."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]

def _unit(v: np.ndarray) -> np.ndarray:
    """Safe unit vector.
    Takes a 3D vector and get its norm (length) - unit vector"""
    n = np.linalg.norm(v) 
    return v / (n + 1e-12) #prevents division by 0

def _triangle_normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray: # maybe change this to consider the central node instead of the first node!!!!!!!!
    """
    Takes the node coordinates (Ca for instance) and return the normal of the plane - a vector perpendicular to the plane - and then normalize it - the unit vector
    """
    return _unit(np.cross(b - a, c - a))

def _sidechain_dirs(ca_list: Sequence[np.ndarray],
                    cb_list: Sequence[np.ndarray]) -> List[np.ndarray]:
    """
    Unit Cα→Cβ vectors per residue.

    NOTE: If you have GLY (no Cβ):
      - skip the triad fir now!!!!!
    """
    out = []
    for ca, cb in zip(ca_list, cb_list):
        out.append(_unit(cb - ca))
    return out

def triad_chirality_with_cb(
    ca_a: np.ndarray, ca_b: np.ndarray, ca_c: np.ndarray,
    cb_a: np.ndarray, cb_b: np.ndarray, cb_c: np.ndarray,
    *,
    weights: Optional[Tuple[float, float, float]] = None,
    outward_normal: Optional[np.ndarray] = None,
    majority_only: bool = True,
) -> Dict[str, Any]:
    """
    Compute a pose-invariant, mirror-variant chirality for a single triad using Cα + Cβ.

    Returns a dict with:
      - chi     : int in {-1,+1}, the handedness bit (flips under mirror, invariant to pose)
      - score   : float in [-1,1], margin = n · s̄  (near 0 ⇒ ambiguous)
      - kappa   : float in (0,1], side-chain consistency (higher = better)
      - n       : np.ndarray(3,), triangle unit normal (from Cα only)
      - sbar    : np.ndarray(3,), final averaged side-chain unit direction
      - details : dict with per-residue signs, which residues contributed, etc.

    Arguments:
      weights        : optional per-residue weights (e.g., RSA, 1/B-factor). If None, all 1.0.
      outward_normal : if provided (pose-invariant local "outward" direction), each side-chain
                       is first oriented outward: s_i <- sign(outward_normal·s_i) * s_i
      majority_only  : if True (default), use only the majority set of side-chains according to
                       sign(n·s_i). This is robust when you have "two outward, one inward".
                       If False, all three side-chains are averaged.
    """
    # Triangle normal from Cα only
    n = _triangle_normal(ca_a, ca_b, ca_c)

    # Side-chain unit directions
    sA, sB, sC = _sidechain_dirs([ca_a, ca_b, ca_c], [cb_a, cb_b, cb_c])
    S = [sA, sB, sC]

    # Optional: orient side-chains outward first - we are not using this 
    if outward_normal is not None:
        u = _unit(outward_normal)
        S = [np.sign(np.dot(u, s)) * s for s in S]

    # Per-residue signs relative to triangle normal
    sigmas = [int(np.sign(np.dot(n, s))) or +1 for s in S]  # treat exact 0 as +1
    maj_sign = +1 if (sigmas.count(+1) >= 2) else -1

    # Which residues to keep in the average?
    if majority_only:
        idx_keep = [i for i, sg in enumerate(sigmas) if sg == maj_sign]  # typically 2 or 3
    else:
        idx_keep = [0, 1, 2]

    # Weights
    W = [1.0, 1.0, 1.0] if weights is None else list(weights)
    sum_w = sum(W[i] for i in idx_keep)

    # Weighted sum of selected side-chains
    vec = np.zeros(3)
    for i in idx_keep:
        vec += W[i] * S[i]

    # Final direction and consistency
    if np.linalg.norm(vec) > 0:
        sbar = _unit(vec)
        kappa = float(np.linalg.norm(vec) / (sum_w + 1e-12))
    else:
        # Degenerate: should not happen unless all S are zero/nan
        sbar = np.array([1.0, 0.0, 0.0])
        kappa = 0.0

    # Margin & bit
    score = float(np.dot(n, sbar))        # ∈ [-1, 1]
    chi   = int(np.sign(score)) or +1     # use +1 when exactly 0

    return {
        "chi": chi,
        "score": score,
        "kappa": kappa,
        "n": n,
        "sbar": sbar,
        "details": {
            "sigmas": sigmas,                 # per-residue sign(n·s_i)
            "kept_indices": idx_keep,         # which residues contributed to s̄
            "maj_sign": maj_sign,
            "weights_used": [W[i] for i in idx_keep],
        },
    }


def collect_all_distances(graph_data, edge_threshold: float) -> np.ndarray:
    """
    Collect all pairwise distances from contact maps of all proteins,
    restricted to 0 < d <= edge_threshold, using upper triangle only.
    """
    all_dists = []

    for gd in graph_data:
        contact_map = gd["contact_map"]
        contact_map = np.asarray(contact_map)
        if contact_map.ndim != 2 or contact_map.shape[0] != contact_map.shape[1]:
            continue  # skip garbage

        iu, ju = np.triu_indices_from(contact_map, k=1)
        dists = contact_map[iu, ju]
        mask = (dists > 0) & (dists <= edge_threshold)
        d_selected = dists[mask]
        if d_selected.size > 0:
            all_dists.append(d_selected)

    if not all_dists:
        return np.array([], dtype=float)

    all_dists = np.concatenate(all_dists)

    return all_dists


def build_dynamic_distance_classes(
    all_dists: np.ndarray,
    n_bins: int,
    edge_threshold: float,
) -> Dict[str, Dict[str, float]]:
    """
    Build dynamic distance classes using quantile-based binning.

    - all_dists: 1D array of distances (already filtered by threshold).
    - n_bins: desired number of bins.
    - edge_threshold: upper cutoff (used as fallback / sanity).

    Returns:
        {
            "bin_1": {"low": low_1, "high": high_1},
            "bin_2": {"low": low_2, "high": high_2},
            ...
        }

    Bins are [low, high] or [low, high) depending on how your find_class works;
    adjust there if needed.
    """
    if all_dists.size == 0:
        return {}

    # Quantile-based edges -> roughly equal number of pairs per bin
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    raw_edges = np.quantile(all_dists, quantiles)

    # Ensure strictly increasing edges (collapse near-duplicates)
    bin_edges = [float(raw_edges[0])]
    for e in raw_edges[1:]:
        e = float(e)
        if e - bin_edges[-1] > 1e-6:
            bin_edges.append(e)

    # If everything collapsed to one value, make a single bin [0, edge_threshold]
    if len(bin_edges) == 1:
        bin_edges = [0.0, float(edge_threshold)]

    # Clamp max edge to edge_threshold (so we never exceed what triads use)
    if bin_edges[-1] > edge_threshold:
        bin_edges[-1] = float(edge_threshold)

    classes = {}
    for i in range(len(bin_edges) - 1):
        low = float(bin_edges[i])
        high = float(bin_edges[i + 1])
        # Skip zero-width bins just in case
        if high <= low:
            continue
        class_name = f"bin_{i+1}"
        classes[class_name] = {"low": low, "high": high}

    return classes


def plot_distance_distribution(
    all_dists: np.ndarray,
    classes: Dict[str, Dict[str, float]],
    output_path: str,
    filename: str = "distance_distribution.png",
):
    """
    Save a histogram of all distances and draw vertical lines at bin boundaries.
    """
    if all_dists.size == 0:
        return

    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, filename)

    plt.figure()
    plt.hist(all_dists, bins=50, density=False, alpha=0.7)
    # Draw bin edges
    edges = sorted(
        {c["low"] for c in classes.values()} |
        {c["high"] for c in classes.values()}
    )
    for e in edges:
        plt.axvline(e, linestyle="--", alpha=0.5)

    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.title("Distance distribution with dynamic bins")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()

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
            chains = config.get("filter_triads_by_chain", None)
            if isinstance(chains, list) and len(chains) > 0:
                u_chain = u.split(":")[0]
                center_chain = center.split(":")[0]
                w_chain = w.split(":")[0]
                if u_chain not in chains and center_chain not in chains and w_chain not in chains:
                    continue

            if u[:5] == w[:5]:
                _u_index, _center_index, _w_index = residue_map[residue_to_tuple(u)], residue_map[residue_to_tuple(center)], residue_map[residue_to_tuple((w))]
                _du = contact_map[_u_index, _center_index]
                _dw = contact_map[_center_index, _w_index]

                if _du <= _dw:
                    outer_sorted = tuple([u, w])
                else:
                    outer_sorted = tuple([w, u])
            else:
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

            u_ca = np.array(G.nodes[outer_sorted[0]].get("ca_coord", (np.nan, np.nan, np.nan)))
            w_ca = np.array(G.nodes[outer_sorted[1]].get("ca_coord", (np.nan, np.nan, np.nan)))
            center_ca = np.array(G.nodes[center].get("ca_coord", (np.nan, np.nan, np.nan)))

            u_cb = np.array(G.nodes[outer_sorted[0]].get("cb_coord", (np.nan, np.nan, np.nan)))
            w_cb = np.array(G.nodes[outer_sorted[1]].get("cb_coord", (np.nan, np.nan, np.nan)))
            center_cb = np.array(G.nodes[center].get("cb_coord", (np.nan, np.nan, np.nan)))


            if not any(np.all(np.isnan(x)) for x in [u_ca, center_ca, w_ca, u_cb, center_cb, w_cb]):
                chi = triad_chirality_with_cb(u_ca, center_ca, w_ca, u_cb, center_cb, w_cb, majority_only=True)["chi"]
            else:
                chi = 0
                print(f"Nan: {u_cb, center_cb, w_cb}")
                print(u, center, w)
                input()
             
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
                    rsa1_class = _as_list(value_to_class(rsa1, config["rsa_bin_width"], config["rsa_filter"]*100, inverse=True, close_tolerance=config["close_tolerance_rsa"]))
                    rsa2_class = _as_list(value_to_class(rsa2, config["rsa_bin_width"], config["rsa_filter"]*100, inverse=True, close_tolerance=config["close_tolerance_rsa"]))
                    rsa3_class = _as_list(value_to_class(rsa3, config["rsa_bin_width"], config["rsa_filter"]*100, inverse=True, close_tolerance=config["close_tolerance_rsa"]))
            else:
                rsa1_class, rsa2_class, rsa3_class = 0, 0, 0
                                            
            dynamic_dist = config.get("dynamic_distance_classes", False)

            if distance_classes is not None and dynamic_dist:
                d1_opts = _as_list(
                    dynamic_distance_to_classes(
                        distance_classes,
                        d1,
                        diff_threshold=config["distance_diff_threshold"],
                        close_tolerance=config["close_tolerance"],
                    )
                )
                d2_opts = _as_list(
                    dynamic_distance_to_classes(
                        distance_classes,
                        d2,
                        diff_threshold=2 * config["distance_diff_threshold"],  # if you want a different rule for d2
                        close_tolerance=config["close_tolerance"],
                    )
                )
                d3_opts = _as_list(
                    dynamic_distance_to_classes(
                        distance_classes,
                        d3,
                        diff_threshold=config["distance_diff_threshold"],
                        close_tolerance=config["close_tolerance"],
                    )
                )
            elif distance_classes is not None:
                # old static-classes behavior, if you already have some find_class()
                d1_opts = _as_list(find_class(distance_classes, d1))
                d2_opts = _as_list(find_class(distance_classes, d2))
                d3_opts = _as_list(find_class(distance_classes, d3))
            else:
                # fallback to fixed-width value_to_class
                d1_opts = _as_list(
                    value_to_class(
                        d1,
                        config["distance_bin_width"],
                        config["edge_threshold"],
                        close_tolerance=config["close_tolerance"],
                    )
                )
                d2_opts = _as_list(
                    value_to_class(
                        d2,
                        config["distance_bin_width"],
                        2 * config["edge_threshold"],
                        close_tolerance=config["close_tolerance"],
                    )
                )
                d3_opts = _as_list(
                    value_to_class(
                        d3,
                        config["distance_bin_width"],
                        config["edge_threshold"],
                        close_tolerance=config["close_tolerance"],
                    )
                )

            # Checagem dos descritores que NÃO são distância (não viram lista)
            # print(config["edge_threshold"],d1_opts, d1, d2_opts, d2, d3_opts, d3)
            prefix = (depth1_class, depth2_class, depth3_class, rsa1_class, rsa2_class, rsa3_class)
            if None in prefix:
                # algum descriptor obrigatório veio inválido; não há o que inserir
                pass
            else:
                # Se qualquer distância não tem classe possível, não gera combinação
                if d1_opts and d2_opts and d3_opts:
                    for d1_c, d2_c, d3_c in product(d1_opts, d2_opts, d3_opts):
                        full_describer = (chi, rsa1_class, rsa2_class, rsa3_class, d1_c, d2_c, d3_c)
                        full_describer_absolute = (chi, rsa1_class, rsa2_class, rsa3_class, d1, d2, d3)

                        triad_class = [u_res_class, center_res_class, w_res_class]
                        triad_abs = [outer_sorted[0], center, outer_sorted[1]]
                        triad_token = (*triad_class, *full_describer)
                        triad_full = (*triad_abs, *full_describer)
                        triad_absolute = (*triad_abs, *full_describer_absolute)
                        if triad_token not in triads:
                            triads[triad_token] = {
                                "count": 1,
                                "triads_full": [triad_full],
                                "triads_absolute": [triad_absolute]
                            }
                        else:
                            triads[triad_token]["count"] += 1
                            triads[triad_token]["triads_full"].append(triad_full)
                            triads[triad_token]["triads_absolute"].append(triad_absolute)
            # else:
                # logging.debug(f"None Found: ({outer_sorted[0], center, outer_sorted[1], d1, d2, d3} | {full_describer}")
    n_triad = 0
    counters = {}
    for triad, data in triads.items():
        n_triad += triads[triad]["count"]

        triads_full = data["triads_absolute"]
        data["triads_absolute_d1"] = sorted(triads_full, key=lambda r: r[-3])

        # if triads[triad]["count"] > 2: print(triad, triads[triad]["count"] )
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

ROLE_TO_PAIR_IDX = {
    "UC": (0, 1),  # (u, c)
    "UW": (0, 2),  # (u, w)
    "CW": (1, 2),  # (c, w)
}

def _build_index(abs_list):
    """
    abs_list: lista de tuplas (r1, r2, r3, d1, d2, d3)
    Retorna estruturas de índice por d1/d2/d3 e vetor de registros.
    """
    records = list(abs_list)  # ids 0..m-1
    by_d1 = sorted((rec[-3], i) for i, rec in enumerate(records))  # (d1, id)
    by_d2 = sorted((rec[-2], i) for i, rec in enumerate(records))  # (d2, id)
    by_d3 = sorted((rec[-1], i) for i, rec in enumerate(records))  # (d3, id)
    return {"records": records, "by_d1": by_d1, "by_d2": by_d2, "by_d3": by_d3}

def _range_ids(sorted_pairs, lo, hi):
    """
    sorted_pairs: lista ordenada de (chave, id)
    retorna conjunto de ids com chave em [lo, hi]
    """
    keys = [k for k, _ in sorted_pairs]
    L = bisect.bisect_left(keys, lo)
    R = bisect.bisect_right(keys, hi)
    # coletar ids só do range
    return {sorted_pairs[j][1] for j in range(L, R)}


def cross_protein_triads(triads_per_protein, diff, check_distances=True):
    """
    triads_per_protein: list of dictionaries per protein.
      For a token t:
        triads_per_protein[p][t]["triads_absolute_d1"]  -> list (r1,r2,r3,d1,d2,d3)
    diff: tolerance for the difference of distances
    """
    import math
    common_tokens = set.intersection(*(set(d.keys()) for d in triads_per_protein))
    cross = {"full": {}} 
    for token in common_tokens:
        triads_len = [len(prot[token]["triads_full"]) for prot in triads_per_protein] if check_distances else [len(prot[token]) for prot in triads_per_protein] 
        prod_triads = math.prod(triads_len)
        if prod_triads > 5000: 
            log.debug(f"Token: {token} | Quantidade em cada proteína: {triads_len} | Combinações totais teórica: {prod_triads}")
    for token in common_tokens:
        if check_distances:
            # obter listas absolutas por proteína para este token
            abs_lists = [prot[token]["triads_absolute_d1"] for prot in triads_per_protein]
            # construir índices por proteína
            idxs = [_build_index(lst) for lst in abs_lists]
            # print(f"idxs: {idxs}")
            sizes = [len(x["records"]) for x in idxs]
            if any(s == 0 for s in sizes):
                continue  

            # escolher a proteína com menor lista como referência
            ref_p = min(range(len(idxs)), key=lambda p: len(idxs[p]["records"]))
            # print(f"Ref_p: {ref_p}")
            ref_idx = idxs[ref_p]
            # print(f"Ref_idx: {ref_idx}")
            other_ps = [p for p in range(len(idxs)) if p != ref_p]

            combos = []  # combinações completas para este token

            for ref_rec in ref_idx["records"]:
                d1_ref, d2_ref, d3_ref = ref_rec[-3], ref_rec[-2], ref_rec[-1]

                w1 = w2 = w3 = diff

                all_candidates = []
                ok = True
                for p in other_ps:
                    idx = idxs[p]
                    cand1 = _range_ids(idx["by_d1"], d1_ref - w1, d1_ref + w1)

                    if not cand1:
                        ok = False
                        break
                    cand2 = _range_ids(idx["by_d2"], d2_ref - w2, d2_ref + w2)

                    if not cand2:
                        ok = False
                        break

                    cand3 = _range_ids(idx["by_d3"], d3_ref - w3, d3_ref + w3)
                    if not cand3:
                        ok = False
                        break

                    cands = cand1 & cand2 & cand3
                    if not cands:
                        ok = False
                        break
     
                    all_candidates.append([idx["records"][i] for i in cands])
                if not ok:
                    continue

                for tail in product(*all_candidates):
                    if ref_p == 0:
                        tup = (ref_rec, *tail)
                    else:
                        # reconstruir na ordem original das proteínas
                        tmp = [None] * len(idxs)
                        tmp[ref_p] = ref_rec
                        k = 0
                        for p in other_ps:
                            tmp[p] = tail[k]
                            k += 1
                        tup = tuple(tmp)
                    combos.append(tup)
        else:
            list_triads_token = [prot[token] for prot in triads_per_protein]
            combos_prod = product(*list_triads_token)
            combos = [tuple(sum(inner, ())) for inner in combos_prod]
        cross["full"][token] = combos

    return cross

def filter_cross_combos(
    cross_combos: Dict,
    graph_datas: List[Dict[str, Any]],
    *,
    distance_std_threshold: float,
    roles_to_check: Iterable[str] = ("UC", "CW"), 
    offset: int = 0,
) -> Dict[Tuple, List[Tuple[Tuple, ...]]]:
    """
    Mantém no combo apenas aqueles em que, para cada 'role' verificado,
    (max(distâncias reais) - min(distâncias reais)) <= distance_std_threshold.

    cross_combos: saída do seu cross_protein_triads (token -> list[combo])
    graph_datas:  lista de graph_data (um por proteína, na MESMA ordem usada no cross_combos)
    """
    roles_to_check = tuple(roles_to_check)
    filtered: Dict[Tuple, List[Tuple[Tuple, ...]]]= {}
    filtered_partial = {}
    cross_combo_len = len(cross_combos["full"].keys())
    compatible = build_token_compatibility(cross_combos["full"])

    full_token = {}
    ti = 1
    # tok_len = "\n".join([f"{token} {len(combos)}" for token, combos in cross_combos["full"].items()])

    for i, (token, combos) in enumerate(cross_combos["full"].items()):
        if i % 100 == 0: 
            print(f"Token: {ti}/{cross_combo_len} | Combos: {len(combos)}")

        full_token[token] = {}
        filtered_partial[token] = set()
        full_partial_tokens = set()
        ti += 1
        kept = []
        tc = 1
        for combo in combos:
            partial_tokens = []
            ok = True
            tc += 1
            for role in roles_to_check:
                _subtoken = []
                vals = []
                for k, tri_k in enumerate(combo):
                    index_graph = k + offset
                    a, b = pair_nodes_from_tri_full(tri_k, role)
                    d = pair_distance_from_graph(a, b, graph_datas[index_graph])
                    _subtoken.append((a, b))
                    vals.append(d)

                sigma = pstdev(vals) if len(vals) > 1 else 0.0
                if sigma > distance_std_threshold:
                    ok = False
                    break
                partial_tokens.append(tuple(_subtoken))
                
            if ok: 
                kept.append(combo)
                full_token[token][combo] = partial_tokens
                full_partial_tokens.union(set(partial_tokens))
        if kept:
            filtered[token] = kept
            filtered_partial[token] = full_partial_tokens

    ti = 1
    for i, (token, combos) in enumerate(filtered.items()):
        if i % 100 == 0:
            print(f"[FILTERED] Token: {ti}/{cross_combo_len} | Combo: {len(combos)}")
        compat_tokens = compatible[token]
        partial_tokens = []
        ti += 1
        kept = []
        tc = 1
        for combo in combos:
            ok = False
            part_tokens = full_token[token][combo]
            for part_token in part_tokens:
                for compat_token in compat_tokens:
                    if part_token in filtered_partial[compat_token]:
                        ok = True
                        break
                if ok:
                    break

            if ok:
                kept.append(combo)

        if kept:
            filtered[token] = kept

    return filtered

def association_tokens_for_combo(
    combo: Tuple[Tuple, ...],
    roles_for_tokens: Iterable[str] = ("UC", "CW"),
) -> List[Tuple]:
    """
    Gera tokens de associação robustos para facilitar a reconstrução depois.
    Formato:
      ("UC", ((i1,j1),(i2,j2),...,(iN,jN)))  com (ik,jk) = índices ordenados do par da proteína k
    """
    roles_for_tokens = tuple(roles_for_tokens)
    toks = []
    for role in roles_for_tokens:
        pairs = tuple(pair_indices_from_tri_full(tri_k, role) for tri_k in combo)
        toks.append((role, pairs))
    return toks


def build_graph_from_filtered_combos(filtered_cross_combos: Dict[Tuple, List[Tuple[Tuple, ...]]]):
    """
    Para cada combo aprovado, cria as arestas (C,U) e (C,W) nos nós alinhados.
    Retorna um set de arestas entre nós-tupla (um rótulo por proteína).
    """
    edges: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()
    for _token, combos in filtered_cross_combos.items():
        for combo in combos:
    
            try:
                U = tuple(tri[0] for tri in combo)
            except Exception as e:
                raise Exception(f"{e}\nCombo: {combo}")
            C = tuple(tri[1] for tri in combo)
            W = tuple(tri[2] for tri in combo)
            edges.add(tuple(sorted((C, U))))
            edges.add(tuple(sorted((C, W))))
    return edges

def rebuild_filtered_combos(filtered_cross_combos: Dict[Tuple, List[Tuple[Tuple, ...]]], graph_nodes):
    filtered_cross_combos_new = {}
    graph_nodes = set(graph_nodes)

    for _token, combos in filtered_cross_combos.items():
        keep = []
        for combo in combos:
            U = tuple(tri[0] for tri in combo)
            C = tuple(tri[1] for tri in combo)
            W = tuple(tri[2] for tri in combo)
            if {U, C, W} <= graph_nodes:
                keep.append(combo)
        if keep:
            filtered_cross_combos_new[_token] = keep
    return filtered_cross_combos_new

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
        # A, B, C, _, _, _, _, _, _, d1, _, d3 = t
        # pats = [
        #     (A,    B,    None,  None,   None,   None,   None,   None,   None,   d1, None,   None),
        #     (B,    A,    None,  None,   None,   None,   None,   None,   None,   d1, None,   None),
        #     (None, A,    B,     None,   None,   None,   None,   None,   None,   None,   None,   d1),
        #     (None, B,    A,     None,   None,   None,   None,   None,   None,   None,   None,   d1),

        #     (C,    B,    None,  None,    None,  None,   None,   None,   None,   d3, None, None),
        #     (B,    C,    None,  None,    None,  None,   None,   None,   None,   d3, None, None),
        #     (None, B,    C,     None,    None,  None,   None,   None,   None,   None, None, d3),
        #     (None, C,    B,     None,    None,  None,   None,   None,   None,   None, None, d3),
        # ]
        A, B, C, *rest = t
        d_uc, d_uw, d_cw = rest[-3], rest[-2], rest[-1]
        pats = [
            (A,    B,    None, d_uc),
            (B,    A,    None, d_uc),
            # (A,    None, B,    d_uw),
            # (B,    None, A,    d_uw),
            (None, B,    C,    d_cw),
            (None, C,    B,    d_cw),
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

def parse_node(node: str) -> Tuple[str, str, int]:
    chain, res, num = node.split(":")
    try:
        num_i = int(num)
    except ValueError:
        import re
        m = re.match(r"(-?\d+)", num)
        if not m:
            raise
        num_i = int(m.group(1))
    return chain, res, num_i

def pair_distance_from_graph(node_a: str, node_b: str, graph_data: Dict[str, Any]) -> float:
    """
    Recalcula a distância contínua a partir de contact_map/residue_map,
    exatamente como no trecho que você mostrou.
    """
    residue_map = graph_data["residue_map_all"]
    contact_map = graph_data["contact_map"]
    a_tuple = residue_to_tuple(node_a)
    b_tuple = residue_to_tuple(node_b)
    ia = residue_map[a_tuple]
    ib = residue_map[b_tuple]
    return float(contact_map[ia, ib])

def pair_nodes_from_tri_full(tri_full: Tuple, role: str) -> Tuple[str, str]:
    i, j = ROLE_TO_PAIR_IDX[role]
    return tri_full[i], tri_full[j]

def pair_indices_from_tri_full(tri_full: Tuple, role: str) -> Tuple[int, int]:
    a, b = pair_nodes_from_tri_full(tri_full, role)
    _, _, ia = parse_node(a)
    _, _, ib = parse_node(b)
    return (ia, ib) if ia < ib else (ib, ia)

def _extract_chain_from_idx(idx, maps):
    res_tuple = maps["residue_maps_unique"].get(idx, None)
    if res_tuple is None:
        return None
    return res_tuple[0]  # cadeia

def _build_threshold_matrix(nodes, maps, threshold_cfg):
    K = len(nodes)
    dim = len(nodes[0])

    # caso seja float, retorna matriz cheia
    if not isinstance(threshold_cfg, dict):
        return np.full((K, K), float(threshold_cfg), dtype=float)

    default_thr = float(threshold_cfg.get("default", 2.0))
    mix_thr = float(threshold_cfg.get("mix", default_thr))

    # chains[p, i] cadeia do no i na proteina p
    chains = np.empty((dim, K), dtype=object)
    for p in range(dim):
        for i in range(K):
            chains[p, i] = _extract_chain_from_idx(nodes[i][p], maps)

    # all_eq True se para todos p a cadeia de i e j sao iguais
    all_eq = np.ones((K, K), dtype=bool)
    for p in range(dim):
        ci = chains[p, :][:, None]  # Kx1
        cj = chains[p, :][None, :]  # 1xK
        all_eq &= (ci == cj)

    # threshold por cadeia
    def thr_for_chain(c):
        return float(threshold_cfg.get(c, default_thr)) if c is not None else default_thr

    row_chain = chains[0, :]                                   # K
    row_thr = np.vectorize(thr_for_chain)(row_chain).astype(float)  # K

    # base com mix e substitui onde all_eq usando broadcast
    T = np.full((K, K), mix_thr, dtype=float)
    T = np.where(all_eq, row_thr[:, None], T)

    return T

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

    mask_invalid_pruned = np.any((stacked_pruned == 0) | np.isnan(stacked_pruned), axis=0)
    mask_invalid_thresh = np.any((stacked_thresh == 0) | np.isnan(stacked_thresh), axis=0)

    stacked_pruned[:, mask_invalid_pruned] = np.nan
    stacked_thresh[:, mask_invalid_thresh] = np.nan

    var_pruned = np.std(stacked_pruned, axis=0)    
    var_thresh = np.std(stacked_thresh, axis=0)  

    mask_pruned = np.any((stacked_pruned == 0) | np.isnan(stacked_pruned), axis=0)
    mask_thresh = np.any((stacked_thresh == 0) | np.isnan(stacked_thresh), axis=0)

    var_pruned = np.where(mask_pruned, np.nan, var_pruned)
    var_thresh = np.where(mask_thresh, np.nan, var_thresh)        

    T = _build_threshold_matrix(nodes, maps, threshold)

    mask_valid = (0 < var_pruned) & (var_pruned < T)
    mask_invalid = ~mask_valid
    var_pruned[mask_valid] = 1
    var_pruned[mask_invalid] = np.nan

    mask_valid = (0 < var_thresh) & (var_thresh < T)
    mask_invalid = ~mask_valid
    var_thresh[mask_valid] = 1
    var_thresh[mask_invalid] = np.nan
 
    new_matrices = {
        "dm_possible_nodes": var_pruned,
        "adj_possible_nodes": var_thresh
    }
    

    return new_matrices, maps

# def execute_step(step, graph_data, graph_collection, max_chunks, qtd_graphs, config, filtered_cross_combos = None):
#     logger = logging.getLogger("association.execute_step")

#     if step == 1:  
#         chunks_triads = [graph_collection["triads"][i:i+max_chunks] for i in range(0, qtd_graphs, max_chunks)]
#         # input(f"Chunks_triads: {len(chunks_triads)}")

#         for i, chunk_triad in enumerate(chunks_triads):

#             logger.info(f"{step} | Creating combos | Chunk: {i}")
#             cross_combos = cross_protein_triads(chunk_triad, config["distance_diff_threshold"])
#             logger.info(f"{step} | Combos created")

#             logger.info(f"{step} | Filtering combos...")
#             qtd_combos_0 = len(cross_combos)
#             filtered_combos = filter_cross_combos(cross_combos, graph_data, distance_std_threshold=config["distance_std_threshold"], roles_to_check=("UC", "CW", "UW"), offset=i*max_chunks)

#             qtd_filtered_combos = len(filtered_combos) - qtd_combos_0
#             logger.info(f"{step} | {qtd_filtered_combos} combos were eliminated")
#     else:
#         if not filtered_cross_combos:
#             raise Exception("Filtered Cross Combos is empty")
#         qtd_graphs = len(filtered_cross_combos)
#         chunk_triads = [filtered_cross_combos[i:i+max_chunks] for i in range(0, qtd_graphs, max_chunks)]
#         filtered_combos_list = [cross_protein_triads(chunk_triad, 1.0, check_distances=False)["full"] for i, chunk_triad in enumerate(chunk_triads) if not print(f"Creating | Combo Chunk: {i}") ] 
#         logger.info(f"{step} | Creating combos | Chunk: {i}")



#     triad_graph_list = [build_graph_from_filtered_combos(filtered_combos) for filtered_combos in filtered_combos_list]
#     tuple_edges_list = [[tuple(edge) for edge in triad_graph] for triad_graph in triad_graph_list]
#     Graphs_list = []
#     for i, tuple_edge in enumerate(tuple_edges_list):
#         G = nx.Graph()
#         G.add_edges_from(tuple_edge)

#         for comp in list(nx.connected_components(G)):
#             if len(comp) == 3:
#                 G.remove_nodes_from(comp)
#         Graphs_list.append((i, G))

#     components_list = [(G[0], list(nx.connected_components(G[1]))) for G in Graphs_list]

#     Graphs_frames = []
#     for comps_idx, components in components_list:
#         index_offset = comps_idx * (max_chunks**passo)
#         comp_id = 1
#         processed_status = 1
#         maps["inv_maps"] = inv_maps
#         components_len = len(components)
#         nodes_qtds = [len(comp) for comp in components]
#         start_frame = time.perf_counter()
#         G = Graphs_list[comps_idx][1]
#         Graphs = [([G], 0)]

#         Frames_comp = set()
#         # input(f"Quantidade de components: {components_len} | Tamanho cada comp: {nodes_qtds}")
#         for component in components:
#             len_component = len(component)
#             log.debug(f"[{passo}] {comp_id}: Processing component {processed_status} / {components_len} with {len_component} nodes")
#             processed_status += 1
#             if len_component <= 4: 
#                 log.debug(f"[{passo}] {comp_id} Skipping component {processed_status-1}, because it has just {len_component} nodes")
#                 continue
#             subG = nx.Graph()
#             subG.add_nodes_from(component)

#             log.debug(f"[{passo}] {comp_id} Adding edges...")
#             for u in component:
#                 for v in G.neighbors(u):
#                     if v in component:
#                         subG.add_edge(u, v)

#             # if len(component) == 346:
#             #     input("Returning component 346")
#             #     return {
#             #         "AssociatedGraph": [([subG], 0)]
#             #     }

#             dm_thresh_graph = np.zeros((metadata["total_length"], metadata["total_length"]))
#             log.debug(f"[{passo}] Creating dm_thresh_graph matrices...")
#             for u, v in subG.edges():
#                 for p, (res_u, res_v) in enumerate(zip(u, v)):
#                     if res_u != res_v:
#                         split_res_u, split_res_v = res_u.split(":"), res_v.split(":")
#                         res_u_tuple = (split_res_u[0], split_res_u[2], split_res_u[1])
#                         res_v_tuple = (split_res_v[0], split_res_v[2], split_res_v[1])
#                         idx_u = inv_maps[p+index_offset][res_u_tuple]
#                         try:
#                             idx_v = inv_maps[p+index_offset][res_v_tuple]
#                         except Exception as e:
#                             raise Exception(f"Problema na proteína {p}, offset: {index_offset}, v tuple: {res_v_tuple}, passo: {passo}, comps_idx: {comps_idx}, comp_id: {comp_id}\nError: {e}")
#                         dm_thresh_graph[idx_u, idx_v] = dm_thresh[idx_u, idx_v]
#                         dm_thresh_graph[idx_v, idx_u] = dm_thresh[idx_v, idx_u]

#             save(f"comp_id_{comp_id}", "dm_thresh_graph", dm_thresh)
#             matrices_dict["dm_thresholded"] = dm_thresh_graph

#             nodes = list(subG.nodes())
#             nodes_indices = []

#             for node in nodes:
#                 node_converted = []
#                 for k, res in enumerate(node):
#                     res_split = res.split(":")
#                     res_tuple = (res_split[0], res_split[2], res_split[1])
#                     res_indice = inv_maps[k+index_offset][res_tuple]
#                     node_converted.append(res_indice)
#                 nodes_indices.append(node_converted)

#             save(f"comp_id_{comp_id}", "nodes_indices", nodes_indices)
#             log.debug(f"[{passo}] {comp_id} Creating the std_matrix...")
#             matrices_mul, maps_mul = create_std_matrix(
#                 nodes=nodes_indices,
#                 matrices=matrices_dict,
#                 maps=maps,
#                 threshold=config["distance_std_threshold"]
#             )

#             log.debug(f"[{passo}] {comp_id} Finished creating the std_matrix.")

#             save(f"comp_id_{comp_id}", "matrices_mul", matrices_mul)
#             save(f"comp_id_{comp_id}", "maps_mul", maps_mul)
#             
#             debugar_ = True if len(nodes) > 300 else False

#             tracer = TraversalTracer(
#                 out_dir="viz_runs",
#                 fmt="mp4",        # ou "mp4"
#                 fps=12,
#                 sample_every=50,  # amostragem para reduzir frames
#                 max_frames=2000,  # corta se passar disso
#                 dpi=110,
#                 enabled=True
#             )

#             if not debugar_: tracer.enabled=False

#             # if passo == 2 and len(nodes) > 5000:
#             #     Graphs.extend([([subG], 1)])
#             #     return {
#             #         "AssociatedGraph": Graphs
#             #     }

#             frames, union_graph, error = generate_frames(
#                 matrices=matrices_mul,
#                 maps=maps_mul,
#                 len_component=len_component,
#                 chunk_id=comps_idx,
#                 step=passo,
#                 debug=debugar_,
#                 tracer=tracer
#             )

#             if len(frames.keys()) > 1:
#                 
#                 if passo < passos: 
#                     Frames_comp = Frames_comp.union(union_graph["edges_residues"])
#                 else:
#                     Graphs.extend([(create_graph(frames, typeEdge="edges_residues", comp_id=comp_id), comp_id)])

#                 comp_id += 1
#             else:
#                 log.debug("Component Refused")
#         tuple_edges_frames = [tuple(edge) for edge in Frames_comp]

#         Graph_union_frames = nx.Graph()
#         Graph_union_frames.add_edges_from(tuple_edges_frames)
#         Graphs_frames.append((comps_idx, Graph_union_frames)) 

#     filtered_cross_combos = [rebuild_filtered_combos(filtered_combos, Graphs_frames[i][1]) for i, filtered_combos in enumerate(filtered_combos_list)]

#     ...


def get_memory_usage_mb():
    """
    Retorna uso de memória RSS em MB, se psutil estiver disponível.
    Caso contrário, retorna None.
    """
    try:
        import psutil
    except ImportError:
        return None

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def write_json_raw(path, data):
    """
    Salva dados em JSON bruto, indentado.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        
def execute_step(
    step_idx: int,
    graph_collection,
    max_chunks: int,
    current_filtered_cross_combos,
    graph_data,
    global_state,
):

    profiling = global_state.get("profiling", None)
    debug = bool(profiling and profiling.get("debug", False))

    if step_idx == 1:
        source = graph_collection["triads"]
    else:
        if not current_filtered_cross_combos:
            raise RuntimeError("current_filtered_cross_combos is empty")
        source = current_filtered_cross_combos

    num_items = len(source)
    chunks = [source[i:i + max_chunks] for i in range(0, num_items, max_chunks)]

    new_filtered_cross_combos = []
    all_step_graphs = []

    if debug:
        step_profile = {
            "step": step_idx,
            "num_items": num_items,
            "num_chunks": len(chunks),
            "chunks": [],       # lista de dicts, um por chunk
            "time_sec": None,   # preenchido no final
            "mem_before_mb": None,
            "mem_after_mb": None,
        }
        step_mem_before = get_memory_usage_mb()
        step_time_start = time.perf_counter()
        print(f"[DEBUG] Step {step_idx} started. Memory: {step_mem_before:.2f} MB")

    else:
        step_profile = None

    for chunk_idx, chunk_triads in enumerate(chunks):
        if debug:
            chunk_time_start = time.perf_counter()
            chunk_mem_before = get_memory_usage_mb()
            print(f"[DEBUG]   Chunk {chunk_idx}: started. "
                  f"Memory before: {chunk_mem_before:.2f} MB. "
                  f"Chunk size: {len(chunk_triads)}")

        rebuilt_combos, final_graphs = process_chunk(
            step_idx=step_idx,
            chunk_idx=chunk_idx,
            chunk_triads=chunk_triads,
            graph_data=graph_data,
            global_state=global_state,
        )

        if isinstance(rebuilt_combos, list):
            new_filtered_cross_combos.extend(rebuilt_combos)
        elif rebuilt_combos is not None:
            new_filtered_cross_combos.append(rebuilt_combos)

        all_step_graphs.extend(final_graphs)

        if debug:
            chunk_time_end = time.perf_counter()
            chunk_mem_after = get_memory_usage_mb()

            print(f"[DEBUG]   Chunk {chunk_idx}: finished. "
                  f"Time: {chunk_time_end - chunk_time_start:.3f}s. "
                  f"Memory after: {chunk_mem_after:.2f} MB.")

            chunk_profile = {
                "chunk_idx": chunk_idx,
                "num_triads_in_chunk": len(chunk_triads),
                "time_sec": chunk_time_end - chunk_time_start,
                "mem_before_mb": chunk_mem_before,
                "mem_after_mb": chunk_mem_after,
            }
            step_profile["chunks"].append(chunk_profile)


    if debug:
        step_time_end = time.perf_counter()
        step_mem_after = get_memory_usage_mb()


        print(f"[DEBUG] Step {step_idx} finished. "
              f"Total time: {step_time_end - step_time_start:.3f}s. "
              f"Final memory: {step_mem_after:.2f} MB\n")

        step_profile["time_sec"] = step_time_end - step_time_start
        step_profile["mem_before_mb"] = step_mem_before
        step_profile["mem_after_mb"] = step_mem_after

        profiling["steps"].append(step_profile)

    return new_filtered_cross_combos, all_step_graphs

def process_chunk(step_idx, chunk_idx, chunk_triads, graph_data, global_state):
    config = global_state["config"]
    dm_thresh = global_state["dm_thresh"]
    inv_maps = global_state["inv_maps"]
    metadata = global_state["metadata"]
    maps = global_state["maps"]
    matrices_dict = global_state["matrices_dict"]
    steps = global_state["steps"]

    max_chunks = config["max_chunks"]

    if step_idx == 1:
        log.debug(f"[step {step_idx}] Creating combos | Chunk: {chunk_idx}")
        cross_combos = cross_protein_triads(chunk_triads, config["distance_diff_threshold"])
        log.info(f"[step {step_idx}] Created {len(cross_combos)}")
        filtered_combos = cross_combos["full"]
        # filtered_combos = filter_cross_combos(
        #     cross_combos,
        #     graph_data,
        #     distance_std_threshold=config["distance_std_threshold"],
        #     roles_to_check=("UC","CW", "UW"),
        #     offset=chunk_idx * max_chunks,
        # )
        # log.info(f"[step {step_idx}] Filtered {len(cross_combos['full'])-len(filtered_combos)} | {len(cross_combos['full'])} | {len(filtered_combos)}")
    else:
        log.debug(f"[step {step_idx}] Creating combos (no distances) | Chunk: {chunk_idx}")
        cross_combos = cross_protein_triads(chunk_triads, config["distance_diff_threshold"], check_distances=False)
        filtered_combos = cross_combos["full"]

    triad_graph = build_graph_from_filtered_combos(filtered_combos)
    tuple_edges = [tuple(edge) for edge in triad_graph]
    G = nx.Graph()
    G.add_edges_from(tuple_edges)

    for comp in list(nx.connected_components(G)):
        if len(comp) == 3:
            G.remove_nodes_from(comp)

    components = list(nx.connected_components(G))

    index_offset_base = chunk_idx * (max_chunks ** step_idx)
    maps["inv_maps"] = inv_maps

    Frames_comp = set()
    final_graphs = []

    comp_id = 1
    processed_status = 1
    num_components = len(components)

    for component in components:
        len_component = len(component)
        log.debug(
            f"[{step_idx}] {comp_id}: Processing component "
            f"{processed_status}/{num_components} with {len_component} nodes"
        )
        processed_status += 1

        if len_component <= 4:
            log.debug(
                f"[{step_idx}] {comp_id} Skipping component "
                f"{processed_status-1}, because it has just {len_component} nodes"
            )
            comp_id += 1
            continue


        subG = G.subgraph(component).copy()

        dm_thresh_graph = np.zeros((metadata["total_length"], metadata["total_length"]))
        log.debug(f"[{step_idx}] Creating dm_thresh_graph matrices...")
        for u, v in subG.edges():
            for p, (res_u, res_v) in enumerate(zip(u, v)):
                if res_u != res_v:
                    split_res_u, split_res_v = res_u.split(":"), res_v.split(":")
                    res_u_tuple = (split_res_u[0], split_res_u[2], split_res_u[1])
                    res_v_tuple = (split_res_v[0], split_res_v[2], split_res_v[1])
                    idx_u = inv_maps[p+index_offset_base][res_u_tuple]
                    try:
                        idx_v = inv_maps[p+index_offset_base][res_v_tuple]
                    except Exception as e:
                        raise Exception(
                            f"Problem at the protein {p}, offset: {index_offset_base}, "
                            f"v tuple: {res_v_tuple}, step: {step_idx}, "
                            f"chunk_idx: {chunk_idx}, comp_id: {comp_id}\nError: {e}"
                        )
                    dm_thresh_graph[idx_u, idx_v] = dm_thresh[idx_u, idx_v]
                    dm_thresh_graph[idx_v, idx_u] = dm_thresh[idx_v, idx_u]

        save(f"comp_id_{comp_id}", "dm_thresh_graph", dm_thresh_graph)
        matrices_dict["dm_thresholded"] = dm_thresh_graph

        nodes = list(subG.nodes())
        nodes_indices = []

        for node in nodes:
            node_converted = []
            for k, res in enumerate(node):
                res_split = res.split(":")
                res_tuple = (res_split[0], res_split[2], res_split[1])
                res_indice = inv_maps[k+index_offset_base][res_tuple]
                node_converted.append(res_indice)
            nodes_indices.append(node_converted)

        save(f"comp_id_{comp_id}", "nodes_indices", nodes_indices)
        log.debug(f"[{step_idx}] {comp_id} Creating the std_matrix...")
        matrices_mul, maps_mul = create_std_matrix(
            nodes=nodes_indices,
            matrices=matrices_dict,
            maps=maps,
            threshold=config["distance_std_threshold"]
        )

        log.debug(f"[{step_idx}] {comp_id} Finished creating the std_matrix.")

        save(f"comp_id_{comp_id}", "matrices_mul", matrices_mul)
        save(f"comp_id_{comp_id}", "maps_mul", maps_mul)
        
        debugar_ = True if len(nodes) > 300 else False

        tracer = TraversalTracer(
            out_dir="viz_runs",
            fmt="mp4",
            fps=12,
            sample_every=50,
            max_frames=2000,
            dpi=110,
            enabled=True
        )

        if not debugar_:
            tracer.enabled = False
        
        end = True if step_idx == steps else False
        frames, union_graph, error = generate_frames(
            component_graph=subG,
            matrices=matrices_mul,
            maps=maps_mul,
            len_component=len_component,
            chunk_id=chunk_idx,
            step=step_idx,
            debug=debugar_,
            tracer=tracer,
            nodes=nodes,
            end=end
        )

        if len(frames.keys()) > 1:
            if step_idx < steps:
                Frames_comp = Frames_comp.union(union_graph["edges_residues"])
            else:
                final_graphs.append(
                    (create_graph(frames, typeEdge="edges_residues", comp_id=comp_id), comp_id)
                )
        else:
            log.debug("Component Refused")

        comp_id += 1

    if step_idx < steps:
        tuple_edges_frames = [tuple(edge) for edge in Frames_comp]
        Graph_union_frames = nx.Graph()
        Graph_union_frames.add_edges_from(tuple_edges_frames)

        rebuilt_combos = rebuild_filtered_combos(filtered_combos, Graph_union_frames)
    else:
        rebuilt_combos = None

    return rebuilt_combos, final_graphs


def association_product(graph_data: list,
                        config: dict,
                        debug: bool = True) -> Union[Dict[str, List], None]:
    logger = logging.getLogger("association.association_product")

    profiling = {
        "debug": debug,
        "steps": [] 
    }
    checks = config.get("checks", {"rsa": True, "depth": True})
    classes = config.get("classes", {})
    max_chunks = config.get("max_chunks", 5)

    dynamic_dist = config.get("dynamic_distance_classes", False)
    distance_bins = config.get("distance_bins", 5)
    edge_threshold = config.get("edge_threshold", 10.0)
    output_path = config.get("output_path", "./outputs")

    # If dynamic distance classes are enabled, build them BEFORE triads
    if dynamic_dist:
        logger.info("Building dynamic distance classes from all contact maps...")
        all_dists = collect_all_distances(graph_data, edge_threshold=edge_threshold)
        if all_dists.size == 0:
            logger.warning("No distances found for dynamic binning; falling back to standard binning.")
        else:
            distance_classes = build_dynamic_distance_classes(
                all_dists=all_dists,
                n_bins=distance_bins,
                edge_threshold=edge_threshold,
            )
            if distance_classes:
                # Attach to classes so find_triads can use it via classes["distance"]
                classes["distance"] = distance_classes
                config["classes"] = classes  # keep config in sync
                logger.info(f"Dynamic distance classes created: {len(distance_classes)} bins.")
                # Optional: plot distribution
                plot_distance_distribution(
                    all_dists=all_dists,
                    classes=distance_classes,
                    output_path=output_path,
                    filename="distance_distribution.png",
                )
                logger.info(f"Distance distribution plot saved to {output_path}/distance_distribution.png")
            else:
                logger.warning("Dynamic distance classes ended up empty; this should not happen.")

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
    
    save("association_product", "graph_collection", graph_collection)
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
                                            distance_threshold=config["edge_threshold"],
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

    save("association_product", "maps", maps)
    save("association_product", "inv_maps", inv_maps)

    for i, graph in enumerate(graph_collection["graphs"]):

        graph_length = len(graph.nodes())
        new_index = current_index + graph_length
        dm_thresh[current_index:new_index, current_index:new_index] = matrices_dict["thresholded_contact_maps"][i]
        dm_prune[current_index:new_index, current_index:new_index] = matrices_dict["pruned_contact_maps"][i]
        current_index = new_index

    save("association_product", "dm_thresh", dm_thresh)
    save("association_product", "dm_prune", dm_prune)
    matrices_dict["dm_thresholded"] = dm_thresh
    matrices_dict["dm_pruned"] = dm_prune

    log.debug("Criando cross combos")
    start_cross = time.perf_counter()

    Graphs = []
    max_chunks = 5

    qtd_graphs = len(graph_collection["graphs"])
    n = qtd_graphs

    steps = 0
    while n > 1:
        n = (n + max_chunks - 1) // max_chunks  
        steps += 1

    filtered_cross_combos = []
    final_graphs = []

    for step_idx in range(1, steps + 1):
        log.debug(f"Executing step {step_idx}")
        filtered_cross_combos, step_graphs = execute_step(
            step_idx,
            graph_collection,
            max_chunks,
            filtered_cross_combos,
            graph_data=graph_data,
            global_state={
                "dm_thresh": dm_thresh,
                "inv_maps": inv_maps,
                "metadata": metadata,
                "maps": maps,
                "matrices_dict": matrices_dict,
                "config": config,
                "steps": steps,
                "profiling": profiling
            }
        )

        if step_idx == steps:
            final_graphs.extend(step_graphs)

    log.debug("Finished creating ALL Frames")
    save("association_product", "Graphs", final_graphs)

    if debug:
        write_json_raw("profiling_report.json", profiling)
        print("[DEBUG] Profiling report saved to association_product/profiling_report.json")

    return {
        "AssociatedGraph": final_graphs
    }         

def generate_frames(component_graph, matrices, maps, len_component, chunk_id, step, debug=False, debug_every=5000, tracer: TraversalTracer=None, nodes=None, end=False):
    """
    Build frames by branching on coherent groups of the frontier.

    dm  : coherence indicator (1 = coherent, else != 1 or NaN)
    adj : adjacency indicator for edges (1 = edge, NaN on diagonal)

    A branch is accepted when the induced subgraph (by adj) over the chosen
    nodes has >= 4 edges and all nodes have degree > 1 within that subgraph.
    """

    dm_raw  = matrices["dm_possible_nodes"].copy()
    adj_raw = matrices["adj_possible_nodes"].copy()
    np.fill_diagonal(dm_raw, 1)
    np.fill_diagonal(adj_raw, np.nan)

    C = (dm_raw == 1)     # Matriz de coerência na forma booleana (os cálculos são mais rápidos)
    A = (adj_raw == 1)    # Matriz de adjacência na forma booleana

    np.fill_diagonal(C, True)
    np.fill_diagonal(A, False)
    K = A.shape[0]

    N_adj = [np.nonzero(A[u])[0] for u in range(K)]  # Vizinhos por adjacência

    if nodes is None:
        nodes = list(component_graph.nodes())

    if len(nodes) != K:
        raise ValueError(
            f"Mismatch between nodes ({len(nodes)}) and matrices size ({K}). "
            "Check node ordering passed to create_std_matrix vs generate_frames."
        )

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    N_adj = [np.nonzero(A[u])[0] for u in range(K)]

    frames = {}

    edges_base = set()
    for u, v in component_graph.edges():
        iu = node_to_idx[u]
        iv = node_to_idx[v]
        if iu == iv:
            continue
        edges_base.add(frozenset((iu, iv)))

    _, edges_idx_0, edges_res_0 = convert_edges_to_residues(edges_base, maps)
    frames[0] = {"edges_indices": edges_idx_0, "edges_residues": edges_res_0}

    def frontier_for(chosen: np.ndarray, inter_mask: np.ndarray) -> tuple:
        """
        A função frontier_for constrói a nova fronteira a partir do conjunto de nós já escolhidos.
        A ideia é a seguinte: dado um conjunto de nós escolhidos (chosen), buscamos todos os vizinhos 
        imediatos desses nós no grafo de adjacência A. Porém, precisamos garantir duas coisas:
            1) Nenhum nó que já foi escolhido pode voltar para a fronteira.
            2) Apenas nós que permanecem coerentes com o inter_mask (máscara de coerência acumulada)
            são mantidos na fronteira.
        O resultado é retornado como uma tupla de índices inteiros representando os nós que formam a nova fronteira.
        """
        if chosen.size == 0:
            # Se não há nós escolhidos, a fronteira é vazia
            return ()
        mask = A[chosen].any(axis=0)   # pega todos os vizinhos dos nós escolhidos
        mask[chosen] = False           # remove os próprios escolhidos da fronteira
        mask &= inter_mask             # mantém apenas nós ainda coerentes
        return tuple(np.nonzero(mask)[0].tolist())  # retorna índices dos vizinhos coerentes


    def valid_subgraph(nodes):
        """
        A função valid_subgraph verifica se um conjunto de nós forma um subgrafo válido.
        As condições para validade são:
            1) O subgrafo induzido deve ter pelo menos 4 arestas.
            2) Todos os nós dentro do subgrafo devem ter grau > 1 (ou seja, sem nós pendurados).
        Caso seja válido, retorna (True, es) onde 'es' é o conjunto de arestas representado por pares de nós.
        Caso contrário, retorna (False, set()).
        """
        nodes = np.asarray(sorted(nodes), dtype=np.int32)  # ordena nós para consistência
        sub = A[np.ix_(nodes, nodes)]  # submatriz de adjacência restrita aos nós escolhidos
        deg = sub.sum(axis=1)          # grau de cada nó dentro do subgrafo

        # Verifica se o número de arestas é suficiente (deg.sum() // 2 = nº de arestas)
        # if int(deg.sum() // 2) < 4:
        #     return False, set()

        if np.any(deg < 1):
            return False, set()

        # Extrai as arestas do subgrafo: usamos apenas a parte triangular superior para evitar duplicatas
        ii, jj = np.where(np.triu(sub, 1))
        es = {frozenset((int(nodes[i]), int(nodes[j]))) for i, j in zip(ii, jj)}

        return True, es
    
    # Bron–Kerbosch com pivô, restrito à fronteira usando a matriz de coerência C
    def maximal_cliques(frontier_set):
        """
        Dado um conjunto de nós da fronteira, queremos dividi-los em grupos
        de nós que são todos coerentes entre si (cliques no grafo de coerência).
        Para isso usamos o algoritmo clássico de Bron–Kerbosch com pivô,
        aplicado no subgrafo de coerência induzido pela fronteira.
        """

        # Transformamos a fronteira (set) em array ordenado, para termos consistência.
        F = np.array(sorted(frontier_set), dtype=np.int32)
        if F.size == 0:
            return []

        # fmask marca quais nós fazem parte da fronteira dentro do universo total K
        fmask = np.zeros(K, dtype=bool)
        fmask[F] = True

        # N é um dicionário de vizinhança no grafo de coerência restrito à fronteira:
        # para cada nó u, guardamos apenas os vizinhos coerentes que também estão na fronteira.
        N = {}
        for u in F:
            # nu = todos os nós coerentes com u (linha de C[u])
            nu = np.flatnonzero(C[u])
            # filtramos para manter só os que também estão na fronteira
            nu = nu[fmask[nu]]
            # salvamos como conjunto, excluindo o próprio u
            N[int(u)] = set(int(x) for x in nu if x != u)

        cliques = []

        def bk(R, P, X):
            """
            Bron–Kerbosch recursivo:
            - R = conjunto atual que estamos construindo (candidato a clique)
            - P = candidatos que ainda podem entrar em R
            - X = candidatos já explorados e que não podem mais entrar
            Quando P e X ficam vazios ao mesmo tempo, R é um clique maximal.
            """
            if not P and not X:
                # Encontramos um clique maximal
                cliques.append(tuple(sorted(R)))
                return

            # Escolhemos um pivô (nó com maior vizinhança em N) para reduzir ramificações
            U = P | X
            pivot = max(U, key=lambda v: len(N[v])) if U else None
            ext = P - (N[pivot] if pivot is not None else set())

            for v in list(ext):
                # Chamamos recursivamente adicionando v ao clique atual
                bk(R | {v}, P & N[v], X & N[v])
                # Após explorar v, movemos ele de P para X (já foi tratado)
                P.remove(v)
                X.add(v)

        # Iniciamos Bron–Kerbosch com:
        # - R vazio (nenhum nó escolhido ainda),
        # - P = todos os nós da fronteira,
        # - X vazio.
        bk(set(), set(int(x) for x in F), set())

        # Ordenamos os cliques encontrados: maiores primeiro
        cliques.sort(key=lambda c: (-len(c), c))
        return cliques


    def mask_signature(mask_bool: np.ndarray) -> bytes:
        # assinatura compacta da máscara boolean
        return np.packbits(mask_bool, bitorder="little").tobytes()

    # expansão unificada: não marca visited aqui
    def expand_groups(chosen, inter_mask, frontier):
        # Achar os grupos de nós coerentes em nossa fronteira é equivalente a converter a nossa fronteira
        # Em um grafo de coerência e então procrurarmos o clique máximo nele
        groups = maximal_cliques(frontier) if frontier else []
        if not groups:
            groups = tuple((v,) for v in sorted(frontier))
        out = []
        for grp in groups:
            inter_g = inter_mask.copy()
            for u in grp:
                inter_g &= C[u]
            chosen_g = tuple(sorted(set(chosen) | set(grp)))
            frontier_g = frontier_for(np.asarray(chosen_g, dtype=np.int32), inter_g)
            sig = (chosen_g, mask_signature(inter_g))
            out.append((sig, chosen_g, inter_g, frontier_g))
        return out

    # ordem de nodes por grau
    deg_all = A.sum(axis=1)
    order_all = np.argsort(-deg_all)
    len_order = len(order_all)
    # order_all = list(nx.algorithms.core.degeneracy_ordering(nx.from_numpy_array(A)))

    seen_edge_sets    = set()   # chaves de arestas aceitas
    visited_states    = set()   # (chosen_tuple, mask_signature)
    checked_node_sets = set()   # frozenset(chosen)
    next_frame_id     = 1

    pushes = pops = accepts = 0
    pruned_dupe_state = 0

    if debug: log.debug("Traversing nodes in descending order of degree ")
    for node_id, node in enumerate(order_all): # Percorremos os nós na ordem do maior grau para o menor grau
        # if tracer and tracer.enabled:
        #     # passa a matriz A do componente que você está explorando aqui; se for o grafo inteiro A, serve
        #     tracer.start_anchor(anchor=int(node), A_sub=A, nodes_map=np.arange(A.shape[0], dtype=int))
        str_node = f"{node_id}/{len_order}"
        if debug: log.debug(f"Node: {str_node}") 
        log.debug(f"[{str_node}] Making copy of inter list...")
        inter = C[int(node)].copy() # Primeiro começamos pegando a linha de coerência que representa o nó e fazemos uma cópia dela (para não modificar o original)
        log.debug(f"[{str_node}] Copy finished")
        accepted = (int(node),) # Iniciamos os nós que serão percorridos com o próprio nó inicial

        neigh = N_adj[int(node)]
        if neigh.size == 0:
            continue

        # filtra vizinhos pela coerência com o node
        # inter[neigh] é bool; pega apenas índices coerentes
        frontier_idx = neigh[inter[neigh]]
        if frontier_idx.size == 0:
            continue
        
        log.debug(f"[{str_node}] Transforming frontier to set...")
        frontier = set(int(v) for v in frontier_idx) # Lista da fronteira 
        log.debug(f"[{str_node}] Finished transforming.")

        # Vamos criar um stack para realizar um busca em profundidade pelo LIFO (Last In First Out)
        stack = []
        stack_small = []

        """
        Sabemos que dado o nó raíz, nem todos os vizinhos são coerentes entre si, então precisamos separar a fronteira em grupos
        de nós que sejam coerentes entre si. Porém, coerência NÃO é transitiva: um nó u pode ser coerente com w e com v, mas w e v
        não serem coerentes entre si. Para cobrir TODAS as possibilidades válidas sem descartar combinações, construímos um “grafo
        de coerência” sobre a fronteira (arestas indicam dm==1 entre pares de vizinhos) e buscamos cliques máximos nele — cada
        clique representa um subconjunto de vizinhos mutualmente coerentes (par a par).
        
        A função expand_groups() encapsula exatamente isso: dado o conjunto 'accepted' (os nós já escolhidos), a máscara 'inter'
        que representa os nós que permanecem coerentes com TODOS os escolhidos, e a 'frontier' atual (vizinhos no grafo de adjacência
        que ainda sobreviveram na máscara), ela:
          1) monta o grafo de coerência restrito à fronteira atual;
          2) extrai cliques máximos (subgrupos de vizinhos mutuamente coerentes);
          3) para cada clique, produz um próximo estado:
             - 'chosen_g'  = accepted ∪ clique (isto é, avançamos escolhendo este subgrupo coerente);
             - 'inter_g'   = inter multiplicado pelo dm de cada nó do clique (filtrando apenas nós coerentes com TODOS os escolhidos + clique);
             - 'frontier_g'= união dos vizinhos de todos os nós em chosen_g, restrita a quem ainda sobrevive em 'inter_g' e que não está em chosen_g.
        
        O retorno é uma lista de assinaturas 'sig' para deduplicar estados (chosen, máscara, fronteira), e as tuplas (chosen_g, inter_g, frontier_g)
        que formam os filhos do estado atual no backtracking. Dessa forma exploramos ramos por “pacotes coerentes”, sem eliminar à força toda a fronteira.
        """
        if debug: log.debug(f"[{str_node}] Creating groups...")
        groups = expand_groups(accepted, inter, frontier)
        if debug: log.debug(f"[{str_node}] Groups created: {len(groups)}")
        # print(groups)
        if debug: log.debug(f"[{str_node}] Adding to tracer...")
        # if tracer and tracer.enabled:
        #     tracer.tick(current=int(node), chosen=accepted, frontier=frontier)

        if debug: log.debug(f"[{str_node}] Adding to stack...")

        for i, (sig, chosen_g, inter_g, frontier_g) in enumerate(groups):
            log.debug(f"[{str_node}] {i}/{len(groups)} group")
            # Evitamos revisitar exatamente o mesmo estado (mesmo conjunto escolhido, mesma máscara e mesma fronteira),
            # o que previne ciclos e explosão desnecessária de busca.
            if sig in visited_states:
                # log.debug(f"[{node_id}/{len_order}] Sig already seen")
                pruned_dupe_state += 1
                continue
            visited_states.add(sig)
            added = tuple(sorted(set(chosen_g) - set(accepted)))

            stack.append((chosen_g, inter_g, frontier_g, added))
            stack_small.append((chosen_g, frontier_g))
            pushes += 1
        if debug: log.debug(f"[{str_node}] Stack created.")
        to_break = False
        # Log inicial para este nó-semente: quantos estados entraram na pilha e um snapshot compacto.
        # log.debug(f"[node {int(node)}] inicializado com {len(stack)} stack | {stack_small}")

        # Loop principal de busca em profundidade (DFS) sobre os estados empilhados.
        current_stack = 0
        while stack:
            current_stack+=1
            str_stack = f"{str_node}|{current_stack}"
            if debug: log.debug(f"[{str_node}] Current stack: {current_stack}")
            chosen_t, inter_m, frontier_t, added_t = stack.pop()
            # if tracer and tracer.enabled:
            #     tracer.tick(current=added_t if added_t else (),
            #                 chosen=chosen_t,
            #                 frontier=frontier_t)
            pops += 1
            chosen = list(chosen_t)
            frontier = set(frontier_t)

            if debug: log.debug(f"[{str_stack}] Expanding group")
            # 1) primeiro tenta expandir (gerar filhos)
            children = []
            groups_s = expand_groups(chosen_t, inter_m, frontier)
            
            if debug: log.debug(f"[{str_stack}] Finished expanding groups: {groups_s} ")

            for sig, c_g, m_g, f_g in groups_s:
                if sig in visited_states:
                    pruned_dupe_state += 1
                    continue
                visited_states.add(sig)
                added = tuple(sorted(set(c_g) - set(chosen_t)))
                children.append((c_g, m_g, f_g, added))

            if children:
                if debug: log.debug(f"[{str_stack}] Found childrens, putting in the stack: {children}")
                # ainda há filhos: empilha e segue a busca
                stack.extend(children)
                pushes += len(children)
            else:
                # 2) nó-folha: avaliamos aceitação do subgrafo
                if debug: log.debug(f"[{str_stack}] Reached the leaf, avaliating...")
                if len(chosen) >= 4:
                    if debug: log.debug(f"[{str_stack}] Len of chose: {len(chosen)}")
                    cn = frozenset(chosen)
                    if cn not in checked_node_sets:
                        checked_node_sets.add(cn)
                        if debug: log.debug(f"[{str_stack}] Checking if it's a valid subraph...")
                        ok, es = valid_subgraph(chosen)
                        if debug: log.debug(f"[{str_stack}] Check finished")
                        if ok:
                            # chave canônica das arestas: pares (u<v), conjunto não-ordenado
                            edge_key = frozenset(tuple(sorted(e)) for e in es)
                            if edge_key not in seen_edge_sets:                       
                                seen_edge_sets.add(edge_key)
                                # if tracer and tracer.enabled:
                                #     tracer.tick(current=None,
                                #                 chosen=chosen,
                                #                 frontier=(),
                                #                 accepted_edges_latest=es)
                                if debug: log.debug(f"[{str_stack}] Converting edges...")
                                _, edges_idx, edges_res = convert_edges_to_residues(es, maps)
                                if debug: log.debug(f"[{str_stack}] Conversion finished.")
                                frames[next_frame_id] = {
                                    "edges_indices": edges_idx,
                                    "edges_residues": edges_res,
                                }
                                accepts += 1
                                next_frame_id += 1

                else:
                    if debug: log.debug(f"Chose have len < 4: {len(chosen)}")

            if (pops % debug_every == 0):
                log.debug(
                    f"[progress] pops={pops:,} pushes={pushes:,} accepts={accepts:,} "
                    f"visited={len(visited_states):,} checked={len(checked_node_sets):,} "
                    f"pruned_dupe={pruned_dupe_state:,} stack={len(stack)}"
                )
                if pushes > 600_000:
                    print(f"The stack is above than 600.000 pushes | {pushes}. The component have {len_component} nodes, and it came from the chunk {chunk_id}, step {step}.")
                    to_break = False
                    # return _, _, True
            if to_break:
                break
        log.debug(f"[{str_node}] Finished while stack.")
        # if tracer and tracer.enabled:
        #     out_file = tracer.end_anchor(name_prefix=f"chunk{chunk_id}_step{step}_anchor{int(node)}")
        #     if debug and out_file:
        #         log.debug(f"[{str_node}][viz] salvo: {out_file}")

    
    if debug: log.debug("Sorting frames...")

    def canon_edge(e):
        """Retorna a aresta em forma canônica (u, v) sem alterar a ordem interna dos nós."""
        u, v = e
        u = tuple(u)
        v = tuple(v)
        return (u, v) if u <= v else (v, u)

    def canon_edges(edges_idx):
        """Retorna conjunto não-direcionado e sem duplicatas de arestas canônicas."""
        return frozenset(canon_edge(e) for e in edges_idx)


    def filter_subframes(frames, debug=False):
        """
        Remove frames que são subconjuntos de outros (mantém apenas os máximos por inclusão),
        mantendo o frame base (id=0) como o primeiro no resultado final.
        """
        if debug: log.debug("Starting subframe filtering...")

        # 1) separar frame base
        base0 = frames.get(0)
        items = [(fid, frames[fid]) for fid in frames if fid != 0]

        if debug:
            log.debug(f"Found {len(items)} non-base frames to process.")
            if base0 is not None:
                log.debug("Frame 0 detected and will be re-added at the end as base frame.")

        # 2) ordenar por tamanho canônico de arestas (decrescente)
        if debug: log.debug("Canonicalizing and sorting frames by edge count...")
        items_sorted = sorted(
            ((fid, fr, canon_edges(fr["edges_indices"])) for fid, fr in items),
            key=lambda t: -len(t[2])
        )

        if debug:
            log.debug("Frames sorted by descending edge count:")
            for idx, (fid, _, es) in enumerate(items_sorted[:10]):
                log.debug(f"  [{idx}] fid={fid} edges={len(es)}")

        # 3) filtro de subconjunto
        kept_sets = []
        kept_frames = []
        total = len(items_sorted)
        for idx, (fid, fr, es) in enumerate(items_sorted):
            if any(es.issubset(k) for k in kept_sets):
                if debug:
                    log.debug(f"[{idx}/{total}] Frame {fid} skipped (subset of existing).")
                continue

            # remove qualquer conjunto estritamente menor já contido nesse novo
            to_drop = [k for k in kept_sets if k.issubset(es) and k != es]
            if to_drop:
                if debug:
                    log.debug(f"[{idx}/{total}] Frame {fid} supersedes {len(to_drop)} smaller frames.")
                kept_sets = [k for k in kept_sets if k not in to_drop]

            kept_sets.append(es)
            kept_frames.append(fr)

            if debug:
                log.debug(f"[{idx}/{total}] Frame {fid} accepted. Total kept: {len(kept_frames)}")

        # 4) montar final_frames com frame 0 na frente
        final_frames = {}
        fid_out = 0
        if base0 is not None:
            final_frames[fid_out] = base0
            fid_out += 1

        for fr in kept_frames:
            final_frames[fid_out] = fr
            fid_out += 1

        if debug:
            log.debug(f"Subframe filtering complete: {len(final_frames)} frames kept total "
                      f"({len(kept_frames)} non-base + base0).")

        return final_frames

    # others = sorted(
    #     (fid for fid in frames if fid != 0),
    #     key=lambda fid: len(frames[fid]["edges_indices"]),
    #     reverse=True
    # )

    # if debug: log.debug("Frames sorted...")
    # ordered = {0: {"edges_indices": [], "edges_residues": []}}
    # for new_id, old_id in enumerate(others, start=1):
    #     ordered[new_id] = frames[old_id]

    # final_frames = {}
    kept_edge_sets = []
    # fid_out = 0
    # if debug: log.debug("Checking frames that are subset of others")
    # len_or = len(ordered.items())
    # for idx, (_, fr) in enumerate(ordered.items()):
    #     if debug: log.debug(f"Checking frame: {idx}/{len_or}")
    #     es = frozenset(map(tuple, fr["edges_indices"]))
    #     if any(es.issubset(k) for k in kept_edge_sets):
    #         continue
    #     # drop strict subsets already kept
    #     if debug: log.debug(f"[{idx}/{len_or}] Dropping subsets...")
    #     to_drop = [k for k in kept_edge_sets if k.issubset(es) and k != es]
    #     if to_drop:
    #         kept_edge_sets = [k for k in kept_edge_sets if k not in to_drop]
    #     if debug: log.debug(f"[{idx}/{len_or}] Subsets dropped.")
    #     final_frames[fid_out] = fr
    #     kept_edge_sets.append(es)
    #     fid_out += 1

    if end:
        final_frames = filter_subframes(frames, debug=debug)
    else:
        final_frames = frames

    # log.debug(f"[done] frames_out={len(final_frames)} "
                # f"(accepted_raw={len(others)}, kept_maximal={len(final_frames)-1})")
    
    edge_to_res = {}          # dict[tuple(edge)] -> tuple(residues) | None
    union_order = []          # ordem determinística: primeira ocorrência ao percorrer os frames
    
    for fid, fr in final_frames.items():
        if fid == 0:
            continue
        ei = fr.get("edges_indices", [])
        er = fr.get("edges_residues", [])
        for j, e in enumerate(ei):
            e_t = e if isinstance(e, tuple) else tuple(e)
            if e_t not in edge_to_res:
                union_order.append(e_t)
                edge_to_res[e_t] = tuple(er[j]) if (er and j < len(er)) else None

    union_edges_indices = union_order
    union_edges_residues = {frozenset(edge_to_res[e]) for e in union_order}

    union_graph = {
        "edges_indices": union_edges_indices,
        "edges_residues": union_edges_residues,
    }

    log.debug(
        f"[union] frames_used={len(final_frames)-1} edges={len(union_edges_indices)} "
        f"(excluded_fid=0)"
    )

    # log.debug(
        # f"[done] frames_out={len(final_frames)} "
        # f"(accepted_raw={len(others)}, kept_maximal={len(final_frames)-1})"
    # )

    return final_frames, union_graph, False

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
    fallback_any_atom: bool = True) -> Tuple[np.ndarray, Dict[Tuple[str, int], int], Dict[Tuple[str, int, str], int]]:
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
    from Bio.PDB import PDBParser, MMCIFParser

    WATER_NAMES = {"HOH", "H2O", "WAT", "TIP3", "SOL"}

    suffix = pdb_file.lower()
    if suffix.endswith((".cif", ".mmcif", ".mcif")):
        parser = MMCIFParser(QUIET=True)
    else:
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
