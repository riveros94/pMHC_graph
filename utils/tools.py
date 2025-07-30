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
from classes.classes import StructureSERD
import matplotlib.pyplot as plt
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

def select_residues_within_range(structureSERD: StructureSERD, range_size=3):
    """
    Seleciona resíduos com numeração próxima ao resíduo selecionado.

    Parameters
    ----------
    structure : Structure
        Instância da classe Structure contendo dados atômicos e de superfície.
    range_size : int, optional
        O intervalo de numeração de resíduos a ser considerado ao redor do resíduo selecionado, por padrão 3.

    Returns
    -------
    list
        Lista de dados de resíduos selecionados com numeração próxima.
    """
    # Calcula a profundidade dos resíduos
    residue_depth = StructureSERD.residue_depth()

    # Converte ResidueNumber para inteiro, se necessário
    residue_depth["ResidueNumber"] = residue_depth["ResidueNumber"].astype(int)

    # Seleciona um resíduo aleatório
    selected_residue = residue_depth.sample(1).iloc[0]
    residue_number = selected_residue["ResidueNumber"]
    chain = selected_residue["Chain"]

    # Seleciona resíduos com numeração próxima
    nearby_residues = residue_depth[
        (residue_depth["ResidueNumber"] >= residue_number - range_size)
        & (residue_depth["ResidueNumber"] <= residue_number + range_size)
        & (residue_depth["Chain"] == chain)
    ]

    # Obtém as coordenadas dos resíduos selecionados
    atomic_data = StructureSERD.atomic
    selected_residues_data = []

    print("Resíduos selecionados e suas profundidades:")

    for _, residue in nearby_residues.iterrows():
        residue_number = int(residue["ResidueNumber"])
        chain = residue["Chain"]

        # Usando numpy para garantir a comparação correta de arrays
        matching_atoms = atomic_data[
            (atomic_data[:, 0] == residue_number) & (atomic_data[:, 1] == chain)
        ]

        if matching_atoms.shape[0] == 0:
            continue  # Se não encontrar o resíduo, ignora

        residue_coords = matching_atoms[:, 4:7].astype(float)

        # Adiciona os dados do resíduo à lista
        selected_residues_data.append({
            "ResidueNumber": residue_number,
            "Chain": chain,
            "ResidueName": residue["ResidueName"],
            "Coordinates": residue_coords.mean(axis=0).tolist(),  # Usando a média das coordenadas dos átomos
            "ResidueDepth": residue["ResidueDepth"]
        })

        # Imprime o número do resíduo e a profundidade
        print(f"Resíduo: {residue_number} | Profundidade: {residue['ResidueDepth']}")

    return selected_residues_data


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
    
def most_frequent_chain(atom_list: List):
    """
    Determine the most frequent chain of atom_list 
    """
    
    chains = [atom.split(':')[0] for atom in atom_list]
    chain_counts = Counter(chains)
    most_common_chain, _ = chain_counts.most_common(1)[0]
    return most_common_chain

def extract_internal_nodes(node: Tuple):
    node_list = []
    
    node_ref = node.pop(-1)
    
    while True: 
        if isinstance(node[0], tuple):
            node_list.append(node.pop(-1))
        elif isinstance(node[0], str):
            node_list.extende([node[0], node[1]])
            break
        else:
            raise f"Unexpected type of node: {node}, type: {type(node)}" 
    
    return node_list, node_ref

def check_identity_same_chain(node_pair: Tuple):
    '''
    Takes a tuple of nodes where each node is a tuple of atoms,
    e.g., (('A:PRO:57', 'A:GLY:58'), 'B:ALA:179'))
    '''
    
    node_list, node_ref = extract_internal_nodes(node_pair) 
    ref_chain = node_ref.split(":")[0] 
    
    chains_node = set([node.split(":")[0] for node in node_list])
    
    if len(chains_node) > 1:
        raise f"There are different chains in the node tuple: {chains_node}"
    
    return "".join(chains_node) == ref_chain


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

def compute_atom_distance(pdb_file, atom_name1, chain_id1, position1, atom_name2, chain_id2, position2):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # Find atoms of the specified residues
    atom1 = None
    atom2 = None

    for model in structure:
        for chain in model:
            chain_id = chain.id
            if chain_id == chain_id1 or chain_id == chain_id2:
                for residue in chain:
                    if (chain_id == chain_id1 and residue.id[1] == position1 and atom_name1 in residue):
                        atom1 = residue[atom_name1]
                    if (chain_id == chain_id2 and residue.id[1] == position2 and atom_name2 in residue):
                        atom2 = residue[atom_name2]
                    if atom1 and atom2:
                        break
                if atom1 and atom2:
                    break

    if not atom1:
        print(f"Warning: 1- Atom {atom_name1} not found for residue at position {position1} in chain {chain_id1}")
        return None
    if not atom2:
        print(f"Warning: 2- Atom {atom_name2} not found for residue at position {position2} in chain {chain_id2}")
        return None

    # Compute Euclidean distance
    distance = atom1 - atom2
    return distance

def convert_node_to_residue(node, residue_maps_unique: Dict):   
    converted_node = tuple(f"{residue_maps_unique[idx][0]}:{residue_maps_unique[idx][2]}:{residue_maps_unique[idx][1]}" for idx in node)
    
    return converted_node

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

def convert_residues_to_indices(residue_edges: List[Tuple[Tuple, Tuple]], residue_maps_unique: Dict) -> List[Tuple[Tuple, Tuple]]:
    """Converte edges no formato de resíduos para índices numéricos.

    Args:
        residue_edges (List[Tuple[Tuple, Tuple]]): Lista de edges no formato de resíduos.
        residue_maps_unique (Dict): Mapeamento de resíduos para índices numéricos.

    Returns:
        List[Tuple[Tuple, Tuple]]: Lista de edges no formato de índices.
    """
    # Criar um dicionário de mapeamento reverso (resíduo -> índice)
    residue_to_index = {
        f"{v[0]}:{v[2]}:{v[1]}": k for k, v in residue_maps_unique.items()
    }
    log.debug(f'Residue to index: {residue_to_index}')
    converted_edges = []
    for (node1, node2) in residue_edges:
        try:
            converted_node1 = tuple(residue_to_index[res] for res in node1 if res in residue_to_index)
            converted_node2 = tuple(residue_to_index[res] for res in node2 if res in residue_to_index)

            if len(converted_node1) == len(node1) and len(converted_node2) == len(node2):
                converted_edges.append((converted_node1, converted_node2))
        except KeyError as e:
            log.error(f"Erro na conversão de resíduos para índices: {e}")

    return converted_edges
    
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
                logger.warning(f"Node '{node}' does not have three parts separated by ':'")
                continue

            chain, res_name, res_num_str = parts
            key = (chain, int(res_num_str), res_name)
            if key in residue_map:
                indices.append(residue_map[key])

        pruned_map = contact_map[np.ix_(indices, indices)]
        np.fill_diagonal(pruned_map, np.nan)
        pruned_contact_maps.append(pruned_map)
        
        thresh_map = pruned_map.copy()
        thresh_map[thresh_map >= distance_threshold] = np.nan
        thresholded_contact_maps.append(thresh_map)
        
        # print(np.info(contact_map))
        thresholded_rsa_maps.append(rsa_map.iloc[indices])
         
        full_res_map = {}
        for i, node in enumerate(nodes):
            parts = node.split(":")
            if len(parts) != 3:
                logger.warning(f"Node '{node}' does not have three parts; skipping for full residue map")
                continue

            chain, res_name, res_num_str = parts
            full_res_map[(chain, int(res_num_str), res_name)] = i
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
    return (res_split[0], int(res_split[2]), res_split[1])

def find_triads(graph_data, association_mode, classes, config, checks):
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

            depth1 = depth.loc[depth["ResNumberChain"] == u_resChain]["ResidueDepth"].values[0]
            depth2 = depth.loc[depth["ResNumberChain"] == center_resChain]["ResidueDepth"].values[0]
            depth3 = depth.loc[depth["ResNumberChain"] == w_resChain]["ResidueDepth"].values[0]

            if checks["depth"]:
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
                
            # if ("A:ARG:163", "C:PRO:4", "C:ILE:5") == (outer_sorted[0], center, outer_sorted[1]):
            #     triad = (u_res_class, center_res_class, w_res_class, *full_describer) 
            #     print(f"{outer_sorted[0], center, outer_sorted[1], d1, d2, d3}")
            #     input(triad)

            
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
                # edges.add(frozenset((center1, u1)))
                # edges.add(frozenset((center1, w1)))
                # edges.add(frozenset((center2, u1)))
                # edges.add(frozenset((center2, w2)))

    # return {"nodes": nodes, "edges": edges}
    return edges

def _combo_label(combo, as_string=True):
    """
    Parameters
    ----------
    combo : tuple  # (triad_p1, triad_p2, …)
    as_string : bool
        • True  → "(A:ALA:4, A:LEU:3, A:LYS:3)(B:…)"  ← human‑readable
        • False → ((A:ALA:4, A:LEU:3, A:LYS:3), (B:…)) ← tuple of tuples

    Returns
    -------
    hashable object that can be used as a NetworkX node id
    """
    triads_no_dist = tuple(triad[:3] for triad in combo)   # drop d1,d2,d3
    if as_string:
        return ''.join(f"({', '.join(triad)})" for triad in triads_no_dist)
    return triads_no_dist            # still hashable (nested tuples)


def dict_to_nx_by_combo(g_dict, string_labels=True):
    """
    Convert the triad‑graph dictionary to a NetworkX graph whose
    *node id* is the association node without distances.

    Parameters
    ----------
    g_dict        : output of build_global_triad_graph_optimised
    string_labels : if True, node ids are human‑readable strings;
                    if False, they are tuples of tuples.

    Returns
    -------
    networkx.Graph
    """
    G = nx.Graph()
    id2label = {}

    # ─── add nodes ────────────────────────────────────────────
    for node in g_dict["nodes"]:
        label = _combo_label(node["combo"], as_string=string_labels)
        id2label[node["id"]] = label
        G.add_node(
            label,
            token=node["token"],
            combo=node["combo"],
            chain_id=_extract_chain_id(node["combo"])
        )

    # ─── add edges ────────────────────────────────────────────
    for edge in g_dict["edges"]:
        u = id2label[edge["node1"]]
        v = id2label[edge["node2"]]
        G.add_edge(u, v, patterns=edge["patterns"])

    return G

def triads_to_dataframe(triads_list: list) -> pd.DataFrame:
    all_keys = set()
    for triads in triads_list:
        all_keys.update(triads.keys())
        
    df = pd.DataFrame(index=sorted(all_keys), columns=range(len(triads_list)))
    
    for i, triads in enumerate(triads_list):
        for triad, data in triads.items():
            df.at[triad, i] = data["count"]

    df = df.fillna(0).astype(int)
    df = df[(df != 0).all(axis=1)]
    
    return df

def _extract_chain_id(combo):
    """
    combo = (triad_p1, triad_p2, …)
    triad_pX = ('A:ALA:4', 'A:LEU:3', 'A:LYS:3', d1, d2, d3)

    Return a string such as 'A' (single protein) or 'A|B|C' (one per protein).
    """
    chains = []
    for triad in combo:
        first_res = triad[0]           # e.g. 'A:ALA:4'
        chain = first_res.split(':')[0]
        chains.append(chain)
    return '|'.join(chains)            # hashable & human‑readable

def create_distance_matrix_final(
    nodes: List[Tuple],
    matrices: dict,
    maps: dict,
    distance_threshold: float,
    std_threshold: float
):
    """
    nodes : list of n-tuples, each (R1, R2, …, Rn) strings "Chain:ResName:ResNum"
    matrices : dict containing:
      • "dm_pruned"      : global M×M array of raw distances
      • "dm_thresholded" : global M×M array, NaN where distance > cutoff
    maps : dict containing:
      • "residue_maps_unique_break": dict k→{ global_idx → (chain, num, resname) }
    distance_threshold : float, max allowed raw distance for dm_possible
    std_threshold : float, max allowed std deviation to keep adj_possible
    """

    dm_pruned    = matrices["dm_pruned"]
    dm_threshold = matrices["dm_thresholded"]

    N = len(nodes)
    F = len(nodes[0])

    dm_possible = np.zeros((N, N), dtype=float)
    dm_std      = np.zeros((N, N), dtype=float)
    adj         = np.zeros((N, N), dtype=float)

    for i, j in combinations(range(N), 2):
        pruned_dists = []
        thresh_dists = []
        ok_pruned    = True

        for k in range(F):
            Ra, Rb = nodes[i][k], nodes[j][k]

            dp = dm_pruned[Ra, Rb]
            dt = dm_threshold[Ra, Rb]

            pruned_dists.append(dp)
            thresh_dists.append(dt)

            # check raw distance threshold
            if np.isnan(dp) or dp > distance_threshold:
                ok_pruned = False

        # compute std dev of raw distances (ignoring NaNs)
        s = float(np.nanstd(pruned_dists, ddof=0))
        dm_std[i, j] = dm_std[j, i] = s

        # mark possible if all raw distances within threshold
        if ok_pruned:
            dm_possible[i, j] = dm_possible[j, i] = 1.0

        # mark adjacency if possible AND all thresholded distances exist AND std within limit
        if ok_pruned and not np.isnan(s) and s <= std_threshold:
            if all(not np.isnan(dt) for dt in thresh_dists):
                adj[i, j] = adj[j, i] = 1.0

    # build return structures
    matrices_out = {
        "dm_possible_nodes":  dm_possible,
        "dm_std":             dm_std,
        "adj_possible_nodes": adj
    }

    maps_out = maps.copy()
    possible = {}
    for idx, node in enumerate(nodes):
        possible[idx]       = node
        possible[str(node)] = idx
    maps_out["possible_nodes"] = possible

    return matrices_out, maps_out

def create_std_matrix(nodes, matrices: dict, maps: dict, threshold: float = 3.0):
    # print(f"Nodes: {nodes}")
    dim = len(nodes[0])
    K = len(nodes) 

    # inv_maps = maps["inv_maps"]

    # idx_lgln = inv_maps[0][("L", 340, "GLN")]
    # idx_dgln = inv_maps[1][("D", 102, "GLN")]

    # idx_lval = inv_maps[0][("L", 342, "VAL")]
    # idx_dval = inv_maps[1][("D", 84, "VAL")]

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
        # print(f"Idx: {idx}")
        # input()
        for i in range(K):
            stacked_pruned[p,  i, :] = matrices["dm_pruned"][idx[i],  idx]
            stacked_thresh[p, i, :]  = matrices["dm_thresholded"][idx[i], idx]

    var_pruned = np.std(stacked_pruned, axis=0)    
    var_thresh = np.std(stacked_thresh, axis=0)  
     
    mask_pruned = np.any((stacked_pruned == 0) | np.isnan(stacked_pruned), axis=0)
    mask_thresh = np.any((stacked_thresh == 0) | np.isnan(stacked_thresh), axis=0)

    var_pruned = np.where(mask_pruned, np.nan, var_pruned)
    var_thresh = np.where(mask_thresh, np.nan, var_thresh)        

    # try:

    #     gln_id = maps["possible_nodes"][f'[{idx_lgln}, {idx_dgln}]']
    #     val_id = maps["possible_nodes"][f'[{idx_lval}, {idx_dval}]']

    #     print(f"Idx_gln: [{idx_lgln}, {idx_dgln}] | {gln_id}")
    #     print(f"Idx_val: [{idx_lval}, {idx_dval}] | {val_id}")

    #     print(stacked_thresh[:, gln_id, val_id])
    #     print(stacked_thresh[:, val_id, gln_id])

    #     print(stacked_thresh[:, gln_id, :])
    #     print(stacked_thresh[:, val_id, :])    

    #     gln_val_adj = var_pruned[gln_id, val_id]
    #     gln_val_dm = var_thresh[gln_id, val_id]

    #     print(matrices["dm_thresholded"][[idx_lgln, idx_dgln], idx_lgln])
    #     print(matrices["dm_pruned"][[idx_lgln, idx_dgln], idx_lgln])

    #     print(matrices["dm_thresholded"][[idx_lval, idx_dval], idx_lval])
    #     print(matrices["dm_pruned"][[idx_lval, idx_dval], idx_lval])

    #     print(gln_val_adj)
    #     print(gln_val_dm)

    #     input()

    # except:
    #     pass

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
                        association_mode: str,
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
    for gd in graph_data:
        df = gd["residue_depth"]
        depth_dict = dict(zip(df["ResNumberChain"], df["ResidueDepth"]))
        depths.append(
            np.array([ depth_dict[node] for node in gd["depth_nodes"] ])
        )

    graph_collection = {
        "graphs": [gd["graph"] for gd in graph_data],
        "triads": [find_triads(gd, association_mode, classes, config, checks) for gd in graph_data],
        "contact_maps": [gd["contact_map"] for gd in graph_data],
        "residue_maps_all": [gd["residue_map_all"] for gd in graph_data],
        "rsa_maps": [gd["rsa"] for gd in graph_data],
        "nodes_graphs": [sorted(list(gd["graph"].nodes())) for gd in graph_data]
    }

    graph_collection["depths_maps"] = depths
 
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

    prot_all_res = np.array([node.split(":")[1]
                            for node_graph in graph_collection["nodes_graphs"]
                            for node in node_graph])
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

    for i, graph in enumerate(graph_collection["graphs"]):

        graph_length = len(graph.nodes())
        new_index = current_index + graph_length
        dm_thresh[current_index:new_index, current_index:new_index] = matrices_dict["thresholded_contact_maps"][i]
        dm_prune[current_index:new_index, current_index:new_index] = matrices_dict["pruned_contact_maps"][i]
        current_index = new_index

    matrices_dict["dm_thresholded"] = dm_thresh
    matrices_dict["dm_pruned"] = dm_prune

    cross_combos = cross_protein_triads(graph_collection["triads"])
    triad_graph = build_combos_graph(cross_combos)
    
    tuple_edges = [tuple(edge) for edge in triad_graph]
    log.debug(f"Number of edges: {len(tuple_edges)}")
    G = nx.Graph()
    G.add_edges_from(tuple_edges)
    components = list(nx.connected_components(G))

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
                    res_u_tuple = (split_res_u[0], int(split_res_u[2]), split_res_u[1])
                    res_v_tuple = (split_res_v[0], int(split_res_v[2]), split_res_v[1])
                    idx_u = inv_maps[p][res_u_tuple]
                    idx_v = inv_maps[p][res_v_tuple]
                    dm_thresh_graph[idx_u, idx_v] = dm_thresh[idx_u, idx_v]
                    dm_thresh_graph[idx_v, idx_u] = dm_thresh[idx_v, idx_u]

        matrices_dict["dm_thresholded"] = dm_thresh_graph
    
        nodes = list(subG.nodes())
        nodes_indices = []

        for node in nodes:
            node_converted = []
            for k, res in enumerate(node):
                res_split = res.split(":")
                res_tuple = (res_split[0], int(res_split[2]), res_split[1])
                res_indice = inv_maps[k][res_tuple]
                node_converted.append(res_indice)
            nodes_indices.append(node_converted)

        matrices_mul, maps_mul = create_std_matrix(
            nodes=nodes_indices,
            matrices=matrices_dict,
            maps=maps,
            threshold=config["distance_diff_threshold"]
        )

        frames = generate_frames(
            matrices=matrices_mul,
            maps=maps_mul,
        )

        if len(frames.keys()) > 1:
            Graphs.extend([(create_graph(frames, typeEdge="edges_residues", comp_id=comp_id), comp_id)])
            comp_id += 1

    return {
                "AssociatedGraph": Graphs
        }


def filter_intermediate_graphs(graphs: list, node_list):
    filtered_graphs = []

    for graph in graphs:
        filtered_graphs.append(nx.subgraph(graph, [node for node in graph.nodes() if node[0] in node_list]))
    
    return filtered_graphs  

def generate_edges(matrices: Dict, maps: Dict, idMatrices: int):
    edges_dict = dict()
    edges_base = set()

    dm_matrix = matrices[idMatrices]["dm_possible_nodes"]
    np.fill_diagonal(dm_matrix, 1)

    adj_matrix = matrices[idMatrices]["adj_possible_nodes"]
    np.fill_diagonal(adj_matrix, np.nan)
    
    lenght = dm_matrix.shape[0]  

    for i in range(0, lenght): 
        adj_indices = np.where(adj_matrix[i] == 1)[0]  
        if len(adj_indices) > 0:
            edges_base.update(map(frozenset, np.column_stack((np.full_like(adj_indices[0], i), adj_indices[0]))))  
   
    edges_base_indices, edges_base_residues = convert_edges_to_residues(edges_base, maps, idMatrices)
    
    edges_dict[0] = {
            "edges_residues": edges_base_residues,
            "edges_indices": edges_base_indices
    } 
 
    return edges_dict

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
        valid_mask = (degrees > 2)
        if valid_mask.sum() < 4:
            continue

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

    # seen = set()
    # for frame in ordered.values():
    #     key = frozenset(map(tuple, frame["edges_indices"]))
    #     assert key not in seen, f"Duplicate frame detected! {key} | {seen}"
    #     seen.add(key)
        
    # final_frames = {}
    # seen = set()
    # i = 0
    # for k, frame in ordered.items():
    #     key = frozenset(map(tuple, frame["edges_indices"]))
    #     if key in seen:
    #         continue
    #     seen.add(key)
    #     final_frames[i] = frame
        
    #     i+= 1
    
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

def generate_frames_old_old(
    matrices: Dict[str, np.ndarray],
    maps: Dict[str, Any]
) -> Dict[int, Dict[str, Set]]:
    """
    Extrai frames de arestas a partir das matrizes filtradas.

    Usa `maps["residue_maps_unique"]` para traduzir cada índice
    diretamente em um Residue = (chain, resnum, resname).
    """

    residue_map     = maps["residue_maps_unique"]  # { idx: (chain, resnum, resname) }
    possible_nodes  = {
        k: tuple(v)
        for k, v in maps["possible_nodes"].items()
        if isinstance(k, int)
    }


    idx_to_node = {
        idx: tuple(residue_map[i] for i in node_tuple)
        for idx, node_tuple in possible_nodes.items()
    }
    node_to_idx = {node: idx for idx, node in idx_to_node.items()}

    dm  = matrices["dm_possible_nodes"].copy()
    adj = matrices["adj_possible_nodes"].copy()
    np.fill_diagonal(dm,  1)
    np.fill_diagonal(adj, np.nan)
    K = dm.shape[0]

    base_edges_res = set()
    for i in range(K):
        for j in np.where(adj[i] == 1)[0]:
            a = idx_to_node[i]
            b = idx_to_node[j]
            base_edges_res.add((a, b))  # mantém direção i→j

    base_edges_idx = {
        (node_to_idx[a], node_to_idx[b])
        for a, b in base_edges_res
    }

    frames = {
        0: {
            "edges_residues": base_edges_res,
            "edges_indices":  base_edges_idx
        }
    }

    seen_edge_sets: Set[frozenset]    = set()
    checked_node_sets: Set[frozenset] = set()
    k = 1

    for i in range(K):
        adj_i = np.where(adj[i] == 1)[0]
        if adj_i.size == 0:
            continue

        inter = dm[i].copy()
        for t in adj_i:
            inter *= dm[t]
        inter_idx = np.where(inter == 1)[0]
        if inter_idx.size < 4:
            continue

        node_set = frozenset(idx_to_node[idx] for idx in inter_idx)
        if node_set in checked_node_sets:
            continue
        checked_node_sets.add(node_set)

        nodes_idx  = np.array([node_to_idx[n] for n in node_set])
        sub_adj    = adj[nodes_idx][:, nodes_idx]
        degrees    = np.sum(np.nan_to_num(sub_adj), axis=1)
        valid_mask = degrees > 2
        if valid_mask.sum() < 4:
            continue

        valid_idx = nodes_idx[valid_mask]
        filtered  = sub_adj[valid_mask][:, valid_mask]

        edges_res = {
            (idx_to_node[int(valid_idx[p])],
             idx_to_node[int(valid_idx[q])])
            for p, q in zip(*np.where(filtered == 1))
        }
        if len(edges_res) < 4 or frozenset(edges_res) in seen_edge_sets:
            continue
        seen_edge_sets.add(frozenset(edges_res))


        edges_idx = {
            (node_to_idx[a], node_to_idx[b])
            for a, b in edges_res
        }
        frames[k] = {
            "edges_residues": edges_res,
            "edges_indices":  edges_idx
        }
        k += 1


    other = sorted(
        (key for key in frames if key != 0),
        key=lambda key: len(frames[key]["edges_indices"]),
        reverse=True
    )
    ordered = {0: frames[0]}
    ordered.update({i + 1: frames[key] for i, key in enumerate(other)})
    
    return ordered

def generate_frames_old(matrices: Dict, maps: Dict):
    """
    Gera arestas usando a matriz filtrada e converte para dois formatos:
    - edges_indices: pares de nós de associação (ex: ((1, 14), (2, 15))).
    - edges_residues: pares de resíduos (ex: ((A:ALA:45, ...), (B:GLY:12, ...))).

    Args:
        distance_matrix_filtered (np.ndarray): Matriz KxK com 1 onde a diferença de distância < cutoff e np.nan caso contrário.
        possible_nodes_map (dict): Dicionário que mapeia índice -> nó de associação.

    Returns:
        edges_indices (List[Tuple]): Lista de arestas em formato de índices.
        edges_residues (List[Tuple]): Lista de arestas convertidas para resíduos.
    """
    frames = dict()
    sets = set()
    checked_sets = set()
    base_frame = set()
    edges_base = set()

    dm_matrix = matrices["dm_possible_nodes"]
    np.fill_diagonal(dm_matrix, 1)

    adj_matrix = matrices["adj_possible_nodes"]
    np.fill_diagonal(adj_matrix, np.nan)
    
    lenght = dm_matrix.shape[0]  

    k = 1
    for i in range(0, lenght):
        adj_indices = np.where(adj_matrix[i] == 1)
        if len(adj_indices) > 0:
            edges_base.update(map(frozenset, np.column_stack((np.full_like(adj_indices[0], i), adj_indices[0]))))  

            intersection = dm_matrix[i]
            
            for target in adj_indices[0]:
                intersection *= dm_matrix[target]
            
            intersection_set = set(np.where(intersection == 1)[0])
        
            if len(intersection_set) > 3 and frozenset(intersection_set) not in checked_sets:
                edges = set()
                checked_sets.add(frozenset(intersection_set)) 
                nodes = np.array(list(intersection_set))

                sub_adj = adj_matrix[nodes][:, nodes]
                
                degrees = np.sum(np.nan_to_num(sub_adj), axis = 1)
                valid_nodes = nodes[degrees > 2]
                valid_nodes_set = frozenset(valid_nodes)
                valid_mask = np.isin(nodes, valid_nodes)
                filtered_sub_adj = sub_adj[valid_mask][:, valid_mask]

                valid_edges = valid_nodes[np.column_stack(np.where(filtered_sub_adj == 1))]

                if len(valid_edges) > 3 and valid_nodes_set not in sets:
                    edges.update(map(frozenset, valid_edges))
            
                    sets.add(valid_nodes_set) 
                    # print(f"Edges: {edges}")
                    edges_indices, edges_residues = convert_edges_to_residues(edges, maps)
                    # print(f"Edges Indices: {edges_indices}")
                    # print(f"Edges Residues: {edges_residues}")
                    # input()
                    frames[k] = {
                            "edges_residues": edges_residues,
                            "edges_indices": edges_indices
                    } 

                    k+= 1
    
    # log.debug(f"Edges Base: {edges_base}")
    edges_base_indices, edges_base_residues = convert_edges_to_residues(edges_base, maps)
    
    frame0 = {
            "edges_residues": edges_base_residues,
            "edges_indices": edges_base_indices
    } 
    

    sorted_items = sorted(frames.items(), key=lambda item: len(item[1]["edges_indices"]), reverse=True)
    frames = {i: frame_data for i, (_, frame_data) in enumerate(sorted_items, start=1)}
    frames[0] = frame0

    log.info(f"I found {len(frames.keys())} frames")
    log.info(f"The base frame has {len(base_frame)} nodes")

    return frames


def generate_frames_old_bkp(matrices: Dict, maps: Dict, idMatrices: int):
    """
    Gera arestas usando a matriz filtrada e converte para dois formatos:
    - edges_indices: pares de nós de associação (ex: ((1, 14), (2, 15))).
    - edges_residues: pares de resíduos (ex: ((A:ALA:45, ...), (B:GLY:12, ...))).

    Args:
        distance_matrix_filtered (np.ndarray): Matriz KxK com 1 onde a diferença de distância < cutoff e np.nan caso contrário.
        possible_nodes_map (dict): Dicionário que mapeia índice -> nó de associação.

    Returns:
        edges_indices (List[Tuple]): Lista de arestas em formato de índices.
        edges_residues (List[Tuple]): Lista de arestas convertidas para resíduos.
    """
    frames = dict()
    sets = set()
    checked_sets = set()
    base_frame = set()
    edges_base = set()

    dm_matrix = matrices[idMatrices]["dm_possible_nodes"]
    np.fill_diagonal(dm_matrix, 1)

    adj_matrix = matrices[idMatrices]["adj_possible_nodes"]
    np.fill_diagonal(adj_matrix, np.nan)
    
    lenght = dm_matrix.shape[0]  

    k = 1
    for i in range(0, lenght):
        adj_indices = np.where(adj_matrix[i] == 1)
        if len(adj_indices) > 0:
            edges_base.update(map(frozenset, np.column_stack((np.full_like(adj_indices[0], i), adj_indices[0]))))  

            intersection = dm_matrix[i]
            
            for target in adj_indices[0]:
                intersection *= dm_matrix[target]
            
            intersection_set = set(np.where(intersection == 1)[0])
        
            if len(intersection_set) > 3 and frozenset(intersection_set) not in checked_sets:
                edges = set()
                checked_sets.add(frozenset(intersection_set)) 
                nodes = np.array(list(intersection_set))

                sub_adj = adj_matrix[nodes][:, nodes]
                
                degrees = np.sum(np.nan_to_num(sub_adj), axis = 1)
                valid_nodes = nodes[degrees > 2]
                valid_nodes_set = frozenset(valid_nodes)
                valid_mask = np.isin(nodes, valid_nodes)
                filtered_sub_adj = sub_adj[valid_mask][:, valid_mask]

                valid_edges = valid_nodes[np.column_stack(np.where(filtered_sub_adj == 1))]

                if len(valid_edges) > 3 and valid_nodes_set not in sets:
                    edges.update(map(frozenset, valid_edges))
            
                    sets.add(valid_nodes_set) 
                    edges_indices, edges_residues = convert_edges_to_residues(edges, maps, idMatrices)
                    
                    frames[k] = {
                            "edges_residues": edges_residues,
                            "edges_indices": edges_indices
                    } 

                    k+= 1
    
    # log.debug(f"Edges Base: {edges_base}")
    edges_base_indices, edges_base_residues = convert_edges_to_residues(edges_base, maps, idMatrices)
    
    frame0 = {
            "edges_residues": edges_base_residues,
            "edges_indices": edges_base_indices
    } 
    

    sorted_items = sorted(frames.items(), key=lambda item: len(item[1]["edges_indices"]), reverse=True)
    frames = {i: frame_data for i, (_, frame_data) in enumerate(sorted_items, start=1)}
    frames[0] = frame0

    log.info(f"I found {len(frames.keys())} frames")
    log.info(f"The base frame has {len(base_frame)} nodes")

    return frames

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


def create_distance_matrix_multiple(
    nodes: np.ndarray,
    idMatrices: int,
    matrices: dict,
    maps: dict,
    connect_thresh: float,
    std_thresh: float
):
    """
    nodes: array shape (K, F), cada linha p é o combo p com índices [i_p^1,...,i_p^F]
           referenciando as F matrizes de distância
    matrices deve conter antes:
      • matrices["dm_pruned_list"]      = [dm_pruned_prot1, ..., dm_pruned_protF]
        (cada dm_pruned_protX: matriz booleana 1/0 de conectividade para proteína X)
      • matrices["dm_thresholded_list"] = [dm_dist_prot1, ..., dm_dist_protF]
        (cada dm_dist_protX: matriz float de distâncias na proteína X)
    connect_thresh: float
      → distância máxima para considerar “conectado” em cada proteína
    std_thresh: float
      → desvio‑padrão máximo permitido para manter adjacência
    """

    K, F = nodes.shape
    pruned_list = matrices["dm_pruned_list"]
    dist_list   = matrices["dm_thresholded_list"]

    # empilha para indexação vetorizada: shape (F, M, M)
    stacked_pruned = np.stack(pruned_list, axis=0).astype(bool)
    stacked_dist   = np.stack(dist_list,    axis=0).astype(float)

    # resultados
    dm_pruned = np.ones((K, K), dtype=bool)
    dm_std    = np.zeros((K, K), dtype=float)

    # apenas p<q
    for p, q in combinations(range(K), 2):
        idxp = nodes[p]  # array de length F
        idxq = nodes[q]

        # distâncias em cada frame
        d_conn = stacked_pruned[np.arange(F), idxp, idxq]   # True/False
        d_dist = stacked_dist  [np.arange(F), idxp, idxq]   # floats

        # pruned só se todos frames conectam E cada distância ≤ connect_thresh
        dm_pruned[p, q] = dm_pruned[q, p] = bool((d_conn) & (d_dist <= connect_thresh)).all()

        # std das distâncias brutas
        dm_std[p, q]    = dm_std[q, p]    = float(np.std(d_dist, ddof=0))

    # diagonal
    np.fill_diagonal(dm_pruned, False)
    np.fill_diagonal(dm_std,    0.0)

    # adjacência final p/q: conectado em todas E desvio abaixo do limiar
    adj = dm_pruned & (dm_std <= std_thresh)

    # escreve de volta em matrices/maps
    matrices[idMatrices] = {}
    maps   [idMatrices] = {}

    # mantemos float(1/0) para compatibilidade
    matrices[idMatrices]["dm_possible_nodes"]  = dm_pruned.astype(float)
    matrices[idMatrices]["dm_thresholded"]     = dm_std
    matrices[idMatrices]["adj_possible_nodes"] = adj.astype(float)

    # reconstruir o map de possíveis nós
    maps[idMatrices]["possible_nodes"] = {}
    for i, row in enumerate(nodes):
        tup = tuple(int(x) for x in row)
        maps[idMatrices]["possible_nodes"][i]   = tup
        maps[idMatrices]["possible_nodes"][tup] = i

    return matrices, maps

def create_possible_nodes_multiple(reference_graph_indices: list, associated_nodes: np.ndarray, range_graph: Tuple, complement_graph_indices: Union[list, None] = None):
    all_possible_nodes = []
    (start, end) = range_graph
    for node in reference_graph_indices:
        connected = associated_nodes[node[0], :].copy()
        for k in range(1, len(node)):
            connected *= associated_nodes[node[k], :]
        
        block_indices = list(np.where(connected[start:end] > 0)[0] + start)
        
        if complement_graph_indices:
            block_indices_filtered = [i for i in block_indices if i in complement_graph_indices]
            block_indices = block_indices_filtered
        
        if block_indices:
            block_elements = [[node], block_indices]

            product = [(*x, y) for x,y in itertools.product(*block_elements)]
            all_possible_nodes.extend(product)    
            # log.debug(f"{i}/{reference_graph} Cartesian product finalized")
    # log.debug(f"All possible nodes: {all_possible_nodes}")
    return np.array(all_possible_nodes)


def create_possible_nodes(reference_graph_indices: list, associated_nodes: np.ndarray, range_graph: Tuple):
    all_possible_nodes = []
    (start, end) = range_graph
    for i in reference_graph_indices:
        block_indices = np.where(associated_nodes[:, i] > 0)[0]


        elements = [index for index in block_indices if start <= index < end]

        if elements:

            block_elements = [[i], elements]
            # log.debug(f"{i}/{reference_graph} Making the cartesian product")
            all_possible_nodes.extend(list(itertools.product(*block_elements)))    
            # log.debug(f"{i}/{reference_graph} Cartesian product finalized")
    # log.debug(f"All possible nodes: {all_possible_nodes}")
    return np.array(all_possible_nodes)

def create_distance_matrix_multiple(nodes, idMatrices: int, matrices: dict, maps: dict, threshold = 3):

     
    def submatrix(matrix, indices):
        return matrix[np.ix_(indices, indices)]
    
    submatrices = []
    submatrices_thresh = []
    
    for column in range(0, nodes.shape[1]):
        indices = nodes[:, column]
        sub = submatrix(matrices["dm_pruned"], indices)
        sub_thresh = submatrix(matrices["dm_thresholded"], indices)

        submatrices.append(sub)
        submatrices_thresh.append(sub_thresh)

    # subA = submatrix(matrices["dm_pruned"], i_indices)
    # subA_thresh = submatrix(matrices["dm_thresholded"], i_indices)
    # subB = submatrix(matrices["dm_pruned"], j_indices)
    # subB_thresh = submatrix(matrices["dm_thresholded"], j_indices)
    
    def make_matrix(matrices, threshold):
        shape = matrices[0].shape
        compat_matrix = np.ones(shape, dtype=float)

        for A,B in combinations(matrices, 2):
            diff = np.abs(A-B)
            mask = diff < threshold

            compat_matrix *= mask.astype(float)
        
        return compat_matrix
    
    matrices[idMatrices]= {} 
    maps[idMatrices] = {}

    matrices[idMatrices]["dm_possible_nodes"] = make_matrix(submatrices, threshold)
    matrices[idMatrices]["adj_possible_nodes"] = make_matrix(submatrices_thresh, threshold)
    
    maps[idMatrices]["possible_nodes"] = {}
    for i, node in enumerate(nodes):
        maps[idMatrices]["possible_nodes"][i] = node
        maps[idMatrices]["possible_nodes"][str(node)] = i
    
    return matrices, maps

def create_distance_matrix(nodes, matrices: dict, maps: dict, threshold = 3):
    i_indices = [node[0] for node in nodes]
    j_indices = [node[1] for node in nodes]
    
    def submatrix(matrix, indices):
        return matrix[np.ix_(indices, indices)]
    
    subA = submatrix(matrices["dm_pruned"], i_indices)
    subA_thresh = submatrix(matrices["dm_thresholded"], i_indices)
    subB = submatrix(matrices["dm_pruned"], j_indices)
    subB_thresh = submatrix(matrices["dm_thresholded"], j_indices)
    
    def make_matrix(subA, subB, threshold):
        M = np.abs(subA - subB)
        
        M[M < threshold] = 1
        M[M >= threshold] = np.nan

        return M

    matrices= {} 
    maps_out = {}

    matrices["dm_possible_nodes"] = make_matrix(subA, subB, threshold)
    matrices["adj_possible_nodes"] = make_matrix(subA_thresh, subB_thresh, threshold)
        
    maps_out["possible_nodes"] = {}
    for i, node in enumerate(nodes):
        maps_out["possible_nodes"][i] = node
        maps_out["possible_nodes"][str(node)] = i
    maps["possible_nodes"] = maps_out["possible_nodes"] 
    return matrices, maps

def build_contact_map(pdb_file):
    from Bio.PDB import PDBParser
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    
    # Lista para armazenar dados de cada resíduo:
    # Cada item é uma tupla: (chain_id, residue_number, residue_name, carbon_vector)
    residues_data = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Monta o identificador do resíduo
                chain_id = chain.id
                res_num = residue.id[1]
                res_name = residue.get_resname()
                
                # Escolhe o átomo: CB se existir, caso contrário CA
                carbon_atom_label = "CB" if residue.has_id("CB") else "CA"
                try:
                    carbon_vector = residue[carbon_atom_label].get_vector()
                except KeyError:
                    continue  # Se faltar o átomo, pula o resíduo
                
                residues_data.append((chain_id, res_num, res_name, carbon_vector))
    
    # Ordena os resíduos de forma determinística: por cadeia, número e nome do resíduo
    residues_data_sorted = sorted(residues_data, key=lambda x: (x[0], x[1], x[2]))
    
    # Separa os dados ordenados
    carbon_atoms = [item[3] for item in residues_data_sorted]
    # Para os mapas, "residue_map" usa (chain, res_num) e "residue_map_all" usa (chain, res_num, res_name)
    residue_map = [(item[0], item[1]) for item in residues_data_sorted]
    residue_map_all = [(item[0], item[1], item[2]) for item in residues_data_sorted]
    
    # Cria dicionários de mapeamento (índice) a partir da ordem ordenada
    residue_map_dict = {tup: i for i, tup in enumerate(residue_map)}
    residue_map_dict_all = {tup: i for i, tup in enumerate(residue_map_all)}
    
    # Calcula a matriz de contatos baseada nos vetores dos átomos de carbono
    num_atoms = len(carbon_atoms)
    contact_map = np.zeros((num_atoms, num_atoms), dtype=float)
    
    for i in range(num_atoms - 1):
        for j in range(i + 1, num_atoms):
            distance = (carbon_atoms[i] - carbon_atoms[j]).norm()
            contact_map[i, j] = distance
            contact_map[j, i] = distance
    
    return contact_map, residue_map_dict, residue_map_dict_all

def find_contact_residues(contact_map, residue_map, residue1, residue2):
    #residue_id1 = residue1
    #residue_id2 = residue2
    #if residue_id1 not in residue_map or residue_id2 not in residue_map:
    #    return "Residue ID not found in the structure."
    
    #residue1_index = residue_map[residue_id1].id[1] - 1
    #residue2_index = residue_map[residue_id2].id[1] - 1
    
    residue1_index = residue_map.get(residue1, None)
    residue2_index = residue_map.get(residue2, None)

    return contact_map[residue1_index, residue2_index]

def coefficient_of_variation(x, y):
    """
    Calculate the coefficient of variation (CV) for a pair of numeric values x and y.
    
    Args:
    x: First numeric value
    y: Second numeric value
    
    Returns:
    CV: Coefficient of variation
    """
    mean = np.mean([x, y])
    std_dev = np.std([x, y])
    cv = (std_dev / mean) * 100
    return cv


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
