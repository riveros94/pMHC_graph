from Bio.PDB import *
import numpy as np
import textdistance
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
import logging, sys
from copy import deepcopy
from scipy.special import factorial
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.cutils.combinations_filter import filtered_combinations
from classes.classes import StructureSERD
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

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
            radius = 1.5  # Definindo um raio para a esfera

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


def generate_edges_old(nodes, distance_matrix, residue_maps_unique, graphs, angle_diff, check_angles):
    """Make edges between associated nodes using dista
    nce_matrix criteria and a filter of cross positions.
    The edges are converted from indices associated nodes to residues associated nodes

    Args:
        nodes (List[Tuple]): A list of all possible associated nodes
        distance_matrix (numpy.ndarray): A distance matrix of each residue in same protein
        residue_maps_unique (dict): A dictionary that retrieve the residue from a given indice.
        This dictionary has residues from all proteins with unique indice for each one

    Returns:
        edges (List[List[Tuple]]): A list of edges to make the associated graph
    """
    edges = []
    # log.info(f"Making edges between nodes ({len(nodes)})")

    nodes = np.array(nodes)
    len_nodes = nodes.shape[0]
  
    log.info("Making combinations with filtering....")

    nodes = np.ascontiguousarray(nodes, dtype=np.int64)
    distance_matrix = np.ascontiguousarray(distance_matrix, dtype=np.float64)
    
    log.debug(f"Shape of nodes: {nodes.shape}")
    log.debug(f"Shape of distance_matrix: {distance_matrix.shape}")
    # Utilize a funcao em Cython para gerar as combinacoes e aplicar os filtros
    # Gera combinações de edges com base na matriz de distância
    edges_indices = filtered_combinations(nodes, distance_matrix)

    # Converte as edges para formato de resíduos
    edges_indices2, edges_residues = convert_edges_to_residues(edges_indices, residue_maps_unique)

    map_residue_edge = {}
    for edge_residue, edge_index in zip(edges_residues, edges_indices2):
        if str(edge_residue) in map_residue_edge.keys():
            print(f"I found edge duplicated: {edge_residue}")
        map_residue_edge[edge_residue] = edge_index

    temp_graph = create_graph(edges_residues)

    if check_angles:
        nodes_filtered_residues = filter_nodes_angle(G=temp_graph, graphs=graphs, angle_diff=angle_diff)
        temp_graph.remove_nodes_from([node for node in temp_graph.nodes if node not in nodes_filtered_residues])

    edges_filtered_residues = list(temp_graph.edges())
    # log.debug(f"map_residue_edge: {map_residue_edge.keys()}")
    # log.debug(f"Filtered edges: {edges_filtered_residues}")

    edges_filtered_indices = []
    for edge in edges_filtered_residues:
        if (edge[0], edge[1]) in map_residue_edge.keys():
            edges_filtered_indices.append(map_residue_edge[(edge[0], edge[1])])
        elif (edge[1], edge[0]) in map_residue_edge.keys():
            edges_filtered_indices.append(map_residue_edge[(edge[1], edge[0])])


    log.info(f"Filtered combinations successfully made! Found {len(edges_filtered_residues)} / {len(edges_filtered_indices)} valid edges.")
    
    return edges_filtered_residues, edges_filtered_indices

def convert_node_to_residue(node, residue_maps_unique: Dict):   
    converted_node = tuple(f"{residue_maps_unique[idx][0]}:{residue_maps_unique[idx][2]}:{residue_maps_unique[idx][1]}" for idx in node)
    
    return converted_node

def convert_edges_to_residues(edges: Set[FrozenSet], maps: Dict, idMap: int) -> Tuple[List, List]:
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

    edges_indices = []
    converted_edges = []
    residue_maps_unique = maps["residue_maps_unique"]
    possible_nodes_map = maps[idMap]["possible_nodes"] 
    for edge in edges:
        edge_list = list(edge)
        node1, node2 = possible_nodes_map[edge_list[0]], possible_nodes_map[edge_list[1]]

        converted_node1 = tuple(f"{residue_maps_unique[idx][0]}:{residue_maps_unique[idx][2]}:{residue_maps_unique[idx][1]}" for idx in node1)
        converted_node2 = tuple(f"{residue_maps_unique[idx][0]}:{residue_maps_unique[idx][2]}:{residue_maps_unique[idx][1]}" for idx in node2)

        converted_node1_indice = tuple(idx for idx in node1)
        converted_node2_indice = tuple(idx for idx in node2)
        if set(converted_node1) != set(converted_node2) and check_cross_positions((converted_node1, converted_node2)):
            edges_indices.append((converted_node1_indice, converted_node2_indice))
            converted_edges.append((converted_node1, converted_node2))
        # else:
        #     log.debug(f'Invalid edge: {edge}, {converted_node1}:{converted_node2}')
    return edges_indices, converted_edges

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
    depths_maps = data["depths_maps"]
    nodes_graphs = data["nodes_graphs"]
    
    maps = {"full_residue_maps": [], "residue_maps_unique": {}}
    pruned_contact_maps = []
    thresholded_contact_maps = []
    thresholded_rsa_maps = []
    thresholded_depth_maps = []
    
    for contact_map, rsa_map, depth_map, residue_map, nodes in zip(
            contact_maps, rsa_maps, depths_maps, residue_maps, nodes_graphs):
        
        depth_map["ResidueNumber"] = depth_map["ResidueNumber"].astype(int)
        depth_map["Chain"] = depth_map["Chain"].astype(str)
        
        indices = []
        indices_depth = []
        
        for node in nodes:
            parts = node.split(":")
            if len(parts) != 3:
                logger.warning(f"Node '{node}' does not have three parts separated by ':'")
                continue

            chain, res_name, res_num_str = parts
            key = (chain, int(res_num_str), res_name)
            if key in residue_map:
                indices.append(residue_map[key])
                indices_depth.append((int(res_num_str), chain))
        
        for i, idx in enumerate(indices_depth):
            residue_in_depth = depth_map[
                (depth_map["ResidueNumber"] == idx[0]) & (depth_map["Chain"] == idx[1])
            ]
            if residue_in_depth.empty:
                logger.warning(f"No matching residue in depth map for index {i}: {idx}")
        
        pruned_map = contact_map[np.ix_(indices, indices)]
        np.fill_diagonal(pruned_map, np.nan)
        pruned_contact_maps.append(pruned_map)
        
        thresh_map = pruned_map.copy()
        thresh_map[thresh_map >= distance_threshold] = np.nan
        thresholded_contact_maps.append(thresh_map)
        
        thresholded_rsa_maps.append(rsa_map[indices])
        
        filtered_depth = depth_map[
            depth_map.apply(lambda row: (row["ResidueNumber"], row["Chain"]) in indices_depth, axis=1)
        ]
        filtered_depth = filtered_depth.set_index(["ResidueNumber", "Chain"]).reindex(indices_depth).reset_index()
        thresholded_depth_maps.append(filtered_depth["ResidueDepth"].to_numpy())
        
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
        matrices_dict["thresholded_depth_maps"] = thresholded_depth_maps
    
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

def discrete_distances(distance, n_divisions, threshold):

    length_division = threshold / n_divisions
    
    if distance <= 0 or distance > threshold:
        return None
    
    class_distance = int((distance - 1e-9) // length_division) + 1
    
    return class_distance

def find_triads(G, contact_map, residue_map, residues_classes, threshold):
    triads = {}
    
    for center in G.nodes():
        neighbors = [n for n in G.neighbors(center) if n != center]
        
        for u, w in combinations(neighbors, 2):
            outer_sorted = tuple(sorted([u, w]))
            u_split, center_split, w_split = outer_sorted[0].split(":"), center.split(":"), outer_sorted[1].split(":")
            u_res, center_res, w_res = u_split[1], center_split[1], w_split[1]
            u_class, center_class, w_class = residues_classes[u_res], residues_classes[center_res], residues_classes[w_res] 
 
            u_tuple, center_tuple, w_tuple = (u_split[0], int(u_split[2]), u_split[1]), (center_split[0], int(center_split[2]), center_split[1]), (w_split[0], int(w_split[2]), w_split[1])

            u_index, center_index, w_index = residue_map[u_tuple], residue_map[center_tuple], residue_map[w_tuple]

            d1 = contact_map[u_index, center_index]
            d2 = contact_map[u_index, w_index]
            d3 = contact_map[center_index, w_index]
        
            d1_discrete = discrete_distances(d1, 5, threshold)
            d2_discrete = discrete_distances(d2, 10, 2*threshold)
            d3_discrete = discrete_distances(d3, 5, threshold)
            
            if None not in [d1_discrete, d2_discrete, d3_discrete]:
                triad = (u_class, center_class, w_class, d1_discrete, d2_discrete, d3_discrete)
                if triad not in triads.keys():
                    triads[triad] = {
                        "count": 1,
                        "triads_full": [(outer_sorted[0], center, outer_sorted[1], d1, d2, d3)]
                    }
                else:
                    triads[triad]["count"] += 1
                    triads[triad]["triads_full"].append((outer_sorted[0], center, outer_sorted[1], d1, d2, d3))
       
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

def association_product(graph_data: list,
                        association_mode: str,
                        config: dict,
                        debug: bool = True) -> Union[Dict, None]:
    logger = logging.getLogger("association.association_product")
    checks = config.get("checks", {"neighbors": True, "rsa": True, "depth": True})
    
    residues_classes = create_residues_classes('resources/atchley_aa.csv', config["residues_similarity_cutoff"])

    graph_collection = {
        "graphs": [gd["graph"] for gd in graph_data],
        "triads": [find_triads(gd["graph"], gd["contact_map"], gd["residue_map_all"], residues_classes, config["centroid_threshold"]) for gd in graph_data],
        "contact_maps": [gd["contact_map"] for gd in graph_data],
        "residue_maps_all": [gd["residue_map_all"] for gd in graph_data],
        "rsa_maps": [gd["rsa"] for gd in graph_data],
        "depths_maps": [gd["residue_depth"] for gd in graph_data],
        "nodes_graphs": [sorted(list(gd["graph"].nodes())) for gd in graph_data]
    }
    
    df = triads_to_dataframe(graph_collection["triads"])
    print(df)
    print(sum(df.prod(axis=1)))
    input("Continuar?")


    total_length = sum(len(g.nodes()) for g in graph_collection["graphs"])
    ranges_graph = indices_graphs(graph_collection["graphs"])
    metadata = {
        "total_length": total_length,
        "ranges_graph": ranges_graph
    }
    logger.debug(f"Total number of nodes: {total_length}")
    
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
    
    # Prepare filtering input; filter_maps_by_nodes now updates matrices_dict and returns maps.
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
    
    for res_map in maps["full_residue_maps"]:
        maps["residue_maps_unique"].update({val + current_value: key for key, val in res_map.items()})
        current_value += len(res_map)
 
    if checks.get("neighbors"):
        logger.info("Creating neighbors vector...")
        neighbors_vec = {
            i: graph_message_passing(graph, 'resources/atchley_aa.csv', use_degree=False, norm_features=False)
            for i, graph in enumerate(graph_collection["graphs"])
        }
        logger.info("Neighbors vector created.")
        matrices_dict["neighbors"] = create_similarity_matrix(
            nodes_graphs=graph_collection["nodes_graphs"],
            metadata=metadata,
            residues_factors=neighbors_vec,
            similarity_cutoff=config["neighbor_similarity_cutoff"]
        )

        logger.info("Neighbors similarity matrix created.")
    
    if checks.get("rsa"):
        logger.info("Creating RSA similarity matrix...")
        matrices_dict["rsa"] = create_similarity_matrix(
            nodes_graphs=graph_collection["nodes_graphs"],
            metadata=metadata,
            residues_factors=matrices_dict["thresholded_rsa_maps"],
            similarity_cutoff=config["rsa_similarity_threshold"],
            mode="1d"
        )
        logger.info("RSA similarity matrix created.")
    
    if checks.get("depth"):
        logger.info("Creating depth similarity matrix...")
        matrices_dict["depth"] = create_similarity_matrix(
            nodes_graphs=graph_collection["nodes_graphs"],
            metadata=metadata,
            residues_factors=matrices_dict["thresholded_depth_maps"],
            similarity_cutoff=config["depth_similarity_threshold"],
            mode="1d"
        )
        logger.info("Depth similarity matrix created.")

    
    def make_associated_matrix(matrices: dict, assoc: np.ndarray, chk: dict) -> np.ndarray:
        if chk.get("neighbors"):
            assoc *= matrices["neighbors"]
        if chk.get("rsa"):
            assoc *= matrices["rsa"]
        if chk.get("depth"):
            assoc *= matrices["depth"]

        return assoc
    
    if association_mode == "identity":
        identity_matrix = np.equal(prot_all_res[:, np.newaxis], prot_all_res).astype(float)
        np.fill_diagonal(identity_matrix, np.nan)
    
        matrices_dict["identity"] = identity_matrix
        logger.info("Identity mode: Creating base associated matrix...")

        matrices_dict["associated"] = make_associated_matrix(matrices_dict, identity_matrix.copy(), checks)

        logger.info("Identity associated matrix created.")

    elif association_mode == "similarity":

        logger.info("Similarity mode: Creating associated matrix...")

        residues_factor = create_residues_factors(graphs=graph_collection["graphs"],
                                                    factors_path=config["factors_path"])

        similarity_matrix = create_similarity_matrix(
            nodes_graphs=graph_collection["nodes_graphs"],
            metadata=metadata,
            residues_factors=residues_factor,
            similarity_cutoff=config["residues_similarity_cutoff"]
        )
        matrices_dict["similarity"] = similarity_matrix
        matrices_dict["associated"] = make_associated_matrix(matrices_dict, similarity_matrix.copy(), checks)

        np.fill_diagonal(matrices_dict["associated"], np.nan)

        logger.info("Similarity associated matrix created.")
     
    current_index = 0

    dm_thresh = np.zeros((metadata["total_length"], metadata["total_length"]))
    dm_prune = np.zeros((metadata["total_length"], metadata["total_length"]))

    for i, graph in enumerate(graph_collection["graphs"]):

        graph_length = len(graph.nodes())
        new_index = current_index + graph_length
        matrices_dict["associated"][current_index:new_index, current_index:new_index] = 0
        dm_thresh[current_index:new_index, current_index:new_index] = matrices_dict["thresholded_contact_maps"][i]
        dm_prune[current_index:new_index, current_index:new_index] = matrices_dict["pruned_contact_maps"][i]
        current_index = new_index

    matrices_dict["dm_thresholded"] = dm_thresh
    matrices_dict["dm_pruned"] = dm_prune

    reference_indicesA = [tuple((i,)) for i in range(*ranges_graph[0])]
    intermediate_graphs = dict()

    for i, range_graph in enumerate(ranges_graph[1:]):
        logger.info(f"Building associated graph between reference and graph {i}")
        possible_nodes = create_possible_nodes_multiple(
            reference_graph_indices=reference_indicesA,
            associated_nodes=matrices_dict["associated"],
            range_graph=range_graph
        )

        if len(possible_nodes) < 1:
            logger.info("No valid association nodes found.")
            return None

        logger.info(f"Found {len(possible_nodes)} possible association nodes.")
        matrices_dict, maps = create_distance_matrix_multiple(
            nodes=possible_nodes,
            idMatrices=i,
            matrices=matrices_dict,
            maps=maps,
            threshold=config["distance_diff_threshold"]
        )
        edges = generate_edges(matrices=matrices_dict,
                            maps=maps,
                            idMatrices=i)
         
        logger.info("Creating intermediate graph")
        intermediate_graph = create_graph(edges) 
        if intermediate_graph is None:
            logger.info("No valid edges found; stopping.")
            return None

        # reference_indices = intermediate_graphs[i]["graph"][0].nodes
        reference_indicesA = sorted({(x[0],) for x in intermediate_graph[0].nodes})
        reference_indicesB = sorted({(x[1],) for x in intermediate_graph[0].nodes})

        # reference_indicesA = [(next(g)[0],) for _, g in itertools.groupby(list(intermediate_graph[0].nodes), key=lambda x: x[0])]
        # reference_indicesB = [(next(g)[1],) for _, g in itertools.groupby(list(intermediate_graph[0].nodes), key=lambda x: x[1])]
        
        intermediate_graphs[i] = {
                "graph": intermediate_graph,
                "edges": edges,
                "matrices": matrices_dict,
                "maps": maps,
                "referenceIndicesA": reference_indicesA,
                "referenceIndicesB": reference_indicesB,
                "idMatrices": i
        }
        
    # 
    # reference_indicesA_last = intermediate_graphs[len(ranges_graph)-2]["referenceIndicesA"]
    # reference_indicesA = intermediate_graphs[len(ranges_graph)-2]["referenceIndicesB"]
    # 
    # for i in range(0, len(ranges_graph)-2):
    #     nodes = list(intermediate_graphs[i]["graph"][0].nodes) 

    #     # selected_nodesB = [node for node in nodes if node[0] in reference_indicesA})       
    #     complement_indices = list({node[1] for node in nodes if node[0] in reference_indicesA_last})
    #     log.debug(f"Complement Indices: {len(complement_indices)}") 
    #     log.debug(f"Number of Nodes: {len(nodes)}")
    #     input()
    #     log.info(f"Building the AssociateGraph with {i}")
    #     possible_nodes = create_possible_nodes_multiple(
    #         reference_graph_indices=reference_indicesA,
    #         associated_nodes=matrices_dict["associated"],
    #         range_graph=ranges_graph[i+1],
    #         complement_graph_indices=complement_indices
    #     )

    #     if len(possible_nodes) < 1:
    #         logger.info("No valid association nodes found.")
    #         return None

    #     logger.info(f"Found {len(possible_nodes)} possible association nodes.")
    #     matrices_dict, maps = create_distance_matrix_multiple(
    #         nodes=possible_nodes,
    #         idMatrices=i,
    #         matrices=matrices_dict,
    #         maps=maps,
    #         threshold=config["distance_diff_threshold"]
    #     )
    #     edges = generate_edges(matrices=matrices_dict,
    #                         maps=maps,
    #                         idMatrices=i)
    #      
    #     logger.info("Creating intermediate graph")
    #     intermediate_graph = create_graph(edges) 
    #     if intermediate_graph is None:
    #         logger.info("No valid edges found; stopping.")
    #         return None
    #     
    #     intermediate_graphs[i].update({
    #         "graph": intermediate_graph,
    #         "edges": edges,
    #         "matrices": matrices_dict,
    #         "maps": maps,
    #         "referenceIndicesA": reference_indicesA,
    #         "idMatrices": i
    #         })
    #     reference_indicesA = list(intermediate_graphs[i]["graph"][0].nodes)
    else:
        final_graph = intermediate_graphs[len(intermediate_graphs)-2]
        print("Number of associated nodes base: ",len(final_graph["graph"][0].nodes))
        # if len(ranges_graph) <= 2:
        #     final_graph = intermediate_graphs[max(intermediate_graphs.keys())]
        # else:
        #     final_graph = intermediate_graphs[max(intermediate_graphs.keys())-1]

        frames = generate_frames(
            matrices=final_graph["matrices"],
            maps=final_graph["maps"],
            idMatrices=final_graph["idMatrices"]
        )

        Graphs = create_graph(frames, typeEdge="edges_residues")
        
        attributes = {
            'maps': maps,
            "matrices_dict": matrices_dict,
            "filter_input": filter_input,
            "graph_collection": graph_collection,
            "intermediate_graphs": intermediate_graphs
        }
        
        return {
                "AssociatedGraph": Graphs,
                **attributes
            }


def filter_intermediate_graphs(graphs: list, node_list):
    filtered_graphs = []

    for graph in graphs:
        filtered_graphs.append(nx.subgraph(graph, [node for node in graph.nodes() if node[0] in node_list]))
    
    return filtered_graphs  

def generate_edges_gpu(matrices: dict, maps: dict, idMatrices: int, use_gpu: bool = True):
    import numpy as np
    if use_gpu:
        import cupy as cp
        xp = cp
    else:
        xp = np

    dm = xp.array(matrices[idMatrices]["dm_possible_nodes"])
    xp.fill_diagonal(dm, 1)

    adj = xp.array(matrices[idMatrices]["adj_possible_nodes"])
    xp.fill_diagonal(adj, xp.nan)

    i_idx, j_idx = xp.where(adj == 1)

    if use_gpu:
        pairs = xp.stack([i_idx, j_idx], axis=1)
        pairs = xp.sort(pairs, axis=1)
        pairs = cp.asnumpy(pairs)
    else:
        pairs = np.stack([i_idx, j_idx], axis=1)
        pairs = np.sort(pairs, axis=1)

    pairs = np.unique(pairs, axis=0)
    edges = {frozenset(pair) for pair in pairs}

    idx, res = convert_edges_to_residues(edges, maps, idMatrices)
    return {0: {"edges_residues": res, "edges_indices": idx}}


def generate_frames_gpu(matrices: dict, maps: dict, idMatrices: int, use_gpu: bool = True):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed
    if use_gpu:
        import cupy as cp
        xp = cp
    else:
        xp = np

    dm = xp.array(matrices[idMatrices]["dm_possible_nodes"])
    xp.fill_diagonal(dm, 1)

    adj = xp.array(matrices[idMatrices]["adj_possible_nodes"])
    xp.fill_diagonal(adj, xp.nan)

    L = dm.shape[0]

    def process_row(i):
        local_edges = set()
        row = adj[i]
        inds = xp.where(row == 1)[0]
        if inds.size:
            row_i = xp.full(inds.shape, i)
            if use_gpu:
                pairs = xp.stack([row_i, inds], axis=1)
                pairs = xp.sort(pairs, axis=1)
                pairs = cp.asnumpy(pairs)
            else:
                pairs = np.stack([row_i, inds], axis=1)
                pairs = np.sort(pairs, axis=1)
            for pair in pairs:
                local_edges.add(frozenset(pair))
            stack = xp.concatenate([dm[i][None, :], dm[inds]], axis=0)
            inter = xp.prod(stack, axis=0)
            inter_set = set(np.where(xp.asnumpy(inter) == 1)[0])
            local_frame = None
            if len(inter_set) > 3:
                nodes = np.array(list(inter_set))
                sub_adj = adj[nodes][:, nodes]
                sub_adj_cpu = cp.asnumpy(sub_adj) if use_gpu else np.array(sub_adj)
                deg = np.sum(np.nan_to_num(sub_adj_cpu), axis=1)
                valid_nodes = nodes[deg > 2]
                valid_nodes_set = frozenset(valid_nodes)
                if valid_nodes.size:
                    mask = np.isin(nodes, valid_nodes)
                    sub_adj_filtered = sub_adj_cpu[mask][:, mask]
                    if sub_adj_filtered.size:
                        valid_pairs = np.stack(np.where(sub_adj_filtered == 1), axis=1)
                        if valid_pairs.shape[0] > 3:
                            edges_inner = {frozenset(valid_nodes[pair]) for pair in valid_pairs}
                            idx_inner, res_inner = convert_edges_to_residues(edges_inner, maps, idMatrices)
                            local_frame = (valid_nodes_set, {"edges_residues": res_inner, "edges_indices": idx_inner})
        else:
            local_frame = None
        return local_edges, local_frame

    global_edges = set()
    frames_dict = {}
    unique_frames = set()

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_row, i): i for i in range(L)}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % max(1, L // 100) == 0:
                log.info(f"Processamento das edges: {completed / L * 100:.2f}% concluído")
            local_edges, local_frame = future.result()
            global_edges.update(local_edges)
            if local_frame is not None:
                valid_nodes_set, frame_data = local_frame
                if valid_nodes_set not in unique_frames:
                    unique_frames.add(valid_nodes_set)
                    frames_dict[len(frames_dict) + 1] = frame_data

    if use_gpu:
        i_idx, j_idx = cp.where(adj == 1)
        pairs = cp.stack([i_idx, j_idx], axis=1)
        pairs = cp.sort(pairs, axis=1)
        pairs = cp.asnumpy(cp.unique(pairs, axis=0))
    else:
        i_idx, j_idx = np.where(adj == 1)
        pairs = np.stack([i_idx, j_idx], axis=1)
        pairs = np.sort(pairs, axis=1)
        pairs = np.unique(pairs, axis=0)
    for pair in pairs:
        global_edges.add(frozenset(pair))
    idx_base, res_base = convert_edges_to_residues(global_edges, maps, idMatrices)
    frame0 = {"edges_residues": res_base, "edges_indices": idx_base}

    sorted_frames = sorted(frames_dict.items(), key=lambda item: len(item[1]["edges_indices"]), reverse=True)
    frames = {i: frame_data for i, (_, frame_data) in enumerate(sorted_frames, start=1)}
    frames[0] = frame0

    log.info(f"Total de frames criados: {len(frames.keys())}")

    return frames


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

def generate_frames(matrices: Dict, maps: Dict, idMatrices: int):
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

def create_graph(edges_dict: Dict, typeEdge: str = "edges_indices"):
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
            log.debug(f"Number of nodes graph {k}: {len(G_sub.nodes)}")
            k+= 1

            if k >= 100:
                break
            Graphs.append(G_sub)
    return Graphs

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

def create_distance_matrix(nodes, idMatrices: int, matrices: dict, maps: dict, threshold = 3):
    i_indices, j_indices = nodes[:, 0], nodes[:, 1]
    
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

    matrices[idMatrices]= {} 
    maps[idMatrices] = {}

    matrices[idMatrices]["dm_possible_nodes"] = make_matrix(subA, subB, threshold)
    matrices[idMatrices]["adj_possible_nodes"] = make_matrix(subA_thresh, subB_thresh, threshold)
    
    maps[idMatrices]["possible_nodes"] = {}
    for i, node in enumerate(nodes):
        maps[idMatrices]["possible_nodes"][i] = node
        maps[idMatrices]["possible_nodes"][str(node)] = i
    
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
