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
from typing import Tuple, List, Optional, Union, Dict
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
import networkx as nx
from itertools import combinations

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
        raise f"The node of second graph is a tuple {node_pair}"
    
    second_graph_node = node_pair.pop(-1) 
    node_ref = []
    
    while True:
        if isinstance(node_pair[-1], tuple):
            node_pair = node_pair[0]
        elif isinstance(node_pair[-1], str):
            node_ref = node_pair[-1]
            break
        else:
            raise f"Unexpected type of node. Node: {node_pair[-1]}, type: {type(node_pair[-1])}"
        
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


def check_cross_positions(node_pair_pair: Tuple[Tuple]):
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


def generate_edges(nodes, distance_matrix, residue_maps_unique, graphs, angle_diff):
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

    # Criar grafo temporário para filtrar ângulos
    temp_graph = create_graph(edges_residues)

    # Aplicar a filtragem de ângulos
    nodes_filtered_residues = filter_nodes_angle(G=temp_graph, graphs=graphs, angle_diff=angle_diff)
    temp_graph.remove_nodes_from([node for node in temp_graph.nodes if node not in nodes_filtered_residues])

    # ** Extrair as edges restantes no grafo após a remoção dos nós **
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

def convert_edges_to_residues(edges: List[Tuple], residue_maps_unique: Dict) -> List[Tuple]:
    """Convert the edges that contains tuple of indices to tuple of residues

    Args:
        edges (List[Tuple]): A list that contains tuple of edges that are made of tuples of indices
        residue_maps_unique (Dict): A map that relates the indice to residue

    Returns:
        convert_edge (List[Tuple]): Return edges converted to residues notation
    """
    edges_indices = []
    converted_edges = []

    for edge in edges:
        node1, node2 = edge

        converted_node1 = tuple(f"{residue_maps_unique[idx][0]}:{residue_maps_unique[idx][2]}:{residue_maps_unique[idx][1]}" for idx in node1)
        converted_node2 = tuple(f"{residue_maps_unique[idx][0]}:{residue_maps_unique[idx][2]}:{residue_maps_unique[idx][1]}" for idx in node2)

        if set(converted_node1) != set(converted_node2) and check_cross_positions((converted_node1, converted_node2)):
            edges_indices.append(edge)
            converted_edges.append((converted_node1, converted_node2))
        else:
            log.debug(f'Invalid edge: {edge}, {converted_node1}:{converted_node2}')
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

def filter_reduce_maps(contact_maps: List, rsa_maps: List, residue_maps: List, nodes_graphs: List, depths_maps: List, distance_threshold: float = 10) -> Tuple:
    """Receive a list of contact_maps and filter all values greater than `distance_threshold`. Besides that
    Generate a unique residue map joining residues of all proteins. 

    Args:
        contact_maps (List): List of `contact map` of all proteins
        residue_maps (List): List of `residue map` of all proteins
        nodes_graphs (list): List of nodes in graph
        distance_threshold (float, optional): Threshold to filter big distances. Defaults to 10.

    Returns:
        Tuple: A tuple that contains `filtered contact maps` and `full residues map`
    """
    filtered_contact_maps = []
    filtered_rsa_maps = []
    filtered_depth_maps = []
    for contact_map, rsa_map, depth_map, residue_map, nodes in zip(contact_maps, rsa_maps, depths_maps, residue_maps, nodes_graphs):
        # log.debug(f"Residue Map: {residue_map}")
        depth_map["ResidueNumber"] = depth_map["ResidueNumber"].astype(int)
        depth_map["Chain"] = depth_map["Chain"].astype(str)

        indices = [
            residue_map[tuple([chain, int(res_num), res_name])]
            for chain, res_name, res_num in (node.split(":") for node in nodes)
            if tuple([chain, int(res_num), res_name]) in residue_map
        ]
        
        indices_depth = [
            tuple([int(res_num), chain])
            for chain, res_name, res_num in (node.split(":") for node in nodes)
            if tuple([chain, int(res_num), res_name]) in residue_map
        ]
        
        # log.debug(f"Indices: {indices}")
        # log.debug(f"Indices Depth: {indices_depth}")

        # Validar o alinhamento dos índices com os mapas
        for i, idx in enumerate(indices_depth):
            residue_in_depth_map = depth_map[
                (depth_map["ResidueNumber"] == idx[0]) & (depth_map["Chain"] == idx[1])
            ]
            rsa_value = rsa_map[indices[i]] if i < len(indices) else None

            # log.debug(f"Validation Index {i}: Depth Map Entry: {residue_in_depth_map}")
            # log.debug(f"Validation Index {i}: RSA Value: {rsa_value}")

            if residue_in_depth_map.empty:
                log.warning(f"No matching residue in depth map for index {i}: {idx}")
                # print(f"Index {i} corresponds to Residue {idx}, RSA: {rsa_value}, Depth: {depth_value}")

        # Aplicar os filtros
        filtered_contact_map = contact_map[np.ix_(indices, indices)]
        # log.debug(f"Filtered Contact Map: {filtered_contact_map}")
        filtered_contact_map[filtered_contact_map >= distance_threshold] = 0
        filtered_contact_maps.append(filtered_contact_map)

        filtered_rsa_map = rsa_map[indices]
        filtered_rsa_maps.append(filtered_rsa_map)

        filtered_depth_map = depth_map[
            depth_map.apply(lambda row: (row["ResidueNumber"], row["Chain"]) in indices_depth, axis=1)
        ]
        filtered_depth_map = filtered_depth_map.set_index(["ResidueNumber", "Chain"]).reindex(indices_depth).reset_index()

        # Garantir que apenas a coluna ResidueDepth seja convertida para um array
        filtered_depth_map_array = filtered_depth_map["ResidueDepth"].to_numpy()

        filtered_depth_maps.append(filtered_depth_map_array)

        # Log final para verificação da ordem
        # log.debug(f"Filtered RSA map (Array): {filtered_rsa_map}")
        # log.debug(f"Filtered Depth map (Array): {filtered_depth_map_array}")


    
    full_residue_maps = [
        {(chain, int(res_num), res_name): i for i, (chain, res_name, res_num) in enumerate(
            node.split(":") for node in sorted_nodes
        )}
        for sorted_nodes in nodes_graphs
    ] 
        
    return filtered_contact_maps, filtered_rsa_maps, full_residue_maps, filtered_depth_maps
def indices_graphs(graphs):
    """Make a list that contains indices that indicates the position of each protein in graph

    Args:
        graphs (List): A list of protein's resdiues. Each List has their own residues.

    Returns:
        ranges_graph (List[Tuple]): A list of indicesthat indicates the position of each protein in matrix
    """
    
    lenght_actual = 0
    ranges_graph = []
    for i in range(len(graphs)):
        graph_lenght = len(graphs[i])
        
        new_lenght_actual = lenght_actual + graph_lenght
        ranges_graph.append((lenght_actual, new_lenght_actual))
        lenght_actual = new_lenght_actual
    return ranges_graph
    
def create_similarity_matrix(nodes_graphs: list, ranges_graph: list, total_lenght: int, residues_factors, similarity_cutoff: float = 0.95, mode="dictionary"):
    """Create a similarity matrix using the cosine similiraty between the vectors that represent the factors from each residue.
    The comparassion is made between residues from different proteins

    Args:
        nodes_graphs (List): A list of all protein's nodes
        ranges_graph (List[Tuple]): A list of indices that indicates the position of each protein in matrix
        total_lenght (int): The total number of residues
        neighbors (dict): A dictionary that contains the residue's neighbors of each protein
        similarity_cutoff (float, optional): Similarity cutoff. Defaults to 0.95.

    Returns:
        matrix (np.ndarray): A numpy neigboor's similarity matrix
    """

    matrix = np.zeros((total_lenght, total_lenght))

    for i in range(len(nodes_graphs)):
        for j in range(i+1, len(nodes_graphs)):
            residuesA = residues_factors[i]
            residuesB = residues_factors[j]

            if mode == "dictionary":
                similarities = np.array([cosine_similarity(residuesA[product[0]], residuesB[product[1]]) for product in itertools.product(residuesA, residuesB)])
                similarities = np.reshape(similarities, (len(residuesA), len(residuesB)))
            elif mode == "1d":
                residuesA = residuesA.reshape(-1, 1)
                residuesB = residuesB.reshape(1, -1)
                
                abs_diff = np.abs(residuesA - residuesB)
                
                similarities = 1 - abs_diff
                
            startA, endA, startB, endB = *ranges_graph[i], *ranges_graph[j]
            
            matrix[startA:endA, startB:endB] = similarities
            matrix[startB:endB, startA:endA] = similarities.T

    matrix[matrix < similarity_cutoff] = 0
    matrix[matrix >= similarity_cutoff] = 1
    
    return matrix            

def create_residues_factors(graphs: List, factors_path: str):
    
    read_emb = pd.read_csv(factors_path)
    residue_factors = {}
    
    for i in range(len(graphs)):
        residue_factors[i] = {node: calculate_atchley_average(node, read_emb) for node in graphs[i].nodes}     
    return residue_factors

def association_product(associated_graph_object, graphsList: List, association_mode: str, factors_path: Union[List, None] = None, centroid_threshold: float = 10, residues_similarity_cutoff: float = 0.90, neighbor_similarity_cutoff: float = 0.90, rsa_similarity_threshold: float = 0.90, depth_similarity_threshold: float = 0.90, angle_diff: float = 20, debug: bool = True):
    """Make the associated graph through the cartesian product of graphs, using somem modifications to filter nodes and edges.
    
    Args:
        graphs (List): A list of graphs
        association_mode (str): Association mode of edges
        nodes_graphs (List): A list of nodes graphs
        contact_maps (List): A list of contact maps
        residue_maps_all (List): Full residues maps
        centroid_threshold (float, optional): Threshold to filter big distances. Defaults to 10.
        similarity_cutoff (float, optional): Similarity cutoff. Defaults to 0.95.

    Returns:
        nx.NetworwGraph: The associated graph
    """
   
    graphs = [graph[0] for graph in graphsList]
    contact_maps = [graph[2] for graph in graphsList]
    # residue_maps = [graph[3] for graph in graphsList]
    residue_maps_all = [graph[4] for graph in graphsList]    
    rsa_maps = [graph[5] for graph in graphsList]
    depths_maps = [graph[6] for graph in graphsList]
    nodes_graphs = [list(graph.nodes()) for graph in graphs]

    total_lenght_graphs = sum([len(graph.nodes()) for graph in graphs])

    log.debug(f"Total Lenght Graphs: {total_lenght_graphs}")
    log.info("Creating filtered contact maps and full residue maps...")    
    filtered_contact_maps, filtered_rsa_maps, full_residue_maps, filtered_depth_maps= filter_reduce_maps(contact_maps=contact_maps, rsa_maps=rsa_maps, residue_maps=residue_maps_all, depths_maps=depths_maps, nodes_graphs=nodes_graphs, distance_threshold=centroid_threshold)
    log.info(f"Filtered contact maps and full residue maps created with success!")


    # log.debug(f"Ploting RSA / Depth Correlation")
    # plot_rsa_depth_correlation(filtered_rsa_maps=filtered_rsa_maps, filtered_depth_maps=filtered_depth_maps)

    # input(f"Continuar?")
    prot_all_res = np.array([node.split(":")[1] for node_graph in nodes_graphs for node in node_graph])
    ranges_graph = indices_graphs(graphs)
    
    log.info(f"Creating the Neighbors Vector...")
    neighbors_vec = {i: graph_message_passing(graph, 'resources/atchley_aa.csv', 
                        use_degree=False, 
                        norm_features=False) for i, graph in enumerate(graphs)}
    log.info("Neighbors vector created with success!")

    current_value = 0
    residue_maps_unique = {}
    
    for residue_map in full_residue_maps:
        residue_maps_unique.update({value + current_value: key for key, value in residue_map.items()})
        current_value += len(residue_map)


    log.info("Creating neighbors similarity matrix...")
    neighbors_similarity = create_similarity_matrix(nodes_graphs = nodes_graphs, ranges_graph = ranges_graph, total_lenght = total_lenght_graphs, residues_factors= neighbors_vec, similarity_cutoff = neighbor_similarity_cutoff)
    log.info("Neighbors similarity matrix created with success!")
    
    log.info("Creating RSA similarity matrix...")
    rsa_similarity = create_similarity_matrix(nodes_graphs=nodes_graphs, ranges_graph = ranges_graph, total_lenght= total_lenght_graphs, residues_factors= filtered_rsa_maps, similarity_cutoff=rsa_similarity_threshold, mode="1d")
    log.info("RSA similarity matrix created with success!")

    log.info("Creating Depth similarity matrix...")
    depth_similarity = create_similarity_matrix(nodes_graphs=nodes_graphs, ranges_graph = ranges_graph, total_lenght= total_lenght_graphs, residues_factors= filtered_depth_maps, similarity_cutoff=depth_similarity_threshold, mode="1d")
    log.info("Depth similarity matrix created with success!")
    
    if association_mode == "identity":
        
        identity_matrix = np.equal(prot_all_res[:, np.newaxis], prot_all_res).astype(int)
        np.fill_diagonal(identity_matrix, 0)
        log.info("Identity: Creating associated nodes matrix...")
        associated_nodes_matrix = identity_matrix * neighbors_similarity * rsa_similarity * depth_similarity
        log.info("Identity: Associated nodes matrix created with success!")
        log.debug(f"Dimension of Associated Matrix: {associated_nodes_matrix.shape}")

    elif association_mode == "similarity":
        log.info("Similarity: Creating associated nodes matrix...")
        residues_factors = create_residues_factors(graphs=graphs, factors_path=factors_path)
        similarity_matrix = create_similarity_matrix(nodes_graphs=nodes_graphs, residues_factors=residues_factors, total_lenght=total_lenght_graphs, similarity_cutoff=residues_similarity_cutoff, ranges_graph=ranges_graph)

        associated_nodes_matrix = similarity_matrix * neighbors_similarity * rsa_similarity * depth_similarity
        np.fill_diagonal(associated_nodes_matrix, 0)
        log.info("Similarity: Associated nodes matrix created with success!")
        log.debug(f"Dimension of Associated Matrix: {associated_nodes_matrix.shape}")
    
    lenght_actual = 0
    
    distance_matrix = np.zeros((total_lenght_graphs, total_lenght_graphs))    
    
    for i in range(len(graphs)):
        graph_lenght = len(graphs[i])
        
        new_lenght_actual = lenght_actual + graph_lenght
        
        associated_nodes_matrix[lenght_actual:new_lenght_actual, lenght_actual:new_lenght_actual] = 0
        distance_matrix[lenght_actual:new_lenght_actual, lenght_actual:new_lenght_actual] = filtered_contact_maps[i]
    
        lenght_actual = new_lenght_actual
        
    reference_graph = graphs[0].nodes()
    
    reference_graph_indices = [*range(len(reference_graph))]
    intermediate_edges = []
    intermediate_graphs = []
    log.debug(f"Reference Graph Indices {reference_graph_indices}")
    for i, range_graph in enumerate(ranges_graph[1:]):
        log.info(f"Making associated graph between reference graph and graph {i}")
        possible_nodes = create_possible_nodes(reference_graph_indices=reference_graph_indices, associated_nodes=associated_nodes_matrix, range_graph=range_graph)
        possible_nodes = [node for node in possible_nodes if check_multiple_chains(node, residue_maps_unique)]

        if len(possible_nodes) < 1:
            log.info("There aren't valid association nodes")
            return None
        else:
            log.info(f"Possible nodes: {len(possible_nodes)}")

        intermediate_edges.append(generate_edges(nodes=possible_nodes, distance_matrix=distance_matrix, residue_maps_unique=residue_maps_unique, graphs=[graphs[0], graphs[i+1]], angle_diff=angle_diff))
        intermediate_graph = create_graph(intermediate_edges[i][1])
        intermediate_graphs.append(intermediate_graph)

        if intermediate_graph is None:
            log.info(f"There aren't valid edges in association graph.")
            return None
        
        reference_graph_indices = [next(g)[0] for _, g in itertools.groupby(intermediate_graphs[i].nodes(), key=lambda x:x[0])]
        log.debug(f"Reference Graph Indices {reference_graph_indices}")



    # log.debug(f"Intermediate Graphs: {intermediate_graphs}")
    # filtered_intermediate_graphs = filter_intermediate_graphs(intermediate_graphs[:-1], reference_graph_indices)
    # log.debug(f"Filtered Intermediate Graphs: {filtered_intermediate_graphs}")

    # log.debug(f"Intermediate Edges -1: {intermediate_edges[-1]}")
    Graph = create_graph(intermediate_edges[-1][0])
    # nodes_filtered = filter_nodes_angle(G = Graph, graphs=[graphs[0], graphs[-1]])
    # Graph.remove_nodes_from([node for node in Graph.nodes if node not in nodes_filtered])

    # log.debug(f"Nodes_filtered: {nodes_filtered}")

    attributes = {
        'contact_maps': contact_maps,
        'residue_maps_all': residue_maps_all,
        'rsa_maps': rsa_maps,
        'depths_maps': depths_maps,
        'nodes_graph': nodes_graphs,
        'neighbors_vec': neighbors_vec,
        'residue_maps_unique': residue_maps_unique,
        'filtered_contact_maps': filtered_contact_maps,
        'filtered_rsa_maps': filtered_rsa_maps,
        'full_residue_maps': full_residue_maps,
        'filtered_depth_maps': filtered_depth_maps,
        'neighbors_similarity': neighbors_similarity,
        'rsa_similarity': rsa_similarity,
        'depth_similarity': depth_similarity,
        'associated_nodes_matrix': associated_nodes_matrix,
        'distance_matrix': distance_matrix,
        'identity_matrix': identity_matrix if association_mode == "identity" else None,
        'similarity_matrix': similarity_matrix if association_mode == "similarity" else None,
        'possible_nodes': possible_nodes,
        'intermediate_graphs': intermediate_graphs,
        'intermediate_edges': intermediate_edges
    }

    # Atribuindo os atributos ao objeto de forma dinâmica
    for attr_name, value in attributes.items():
        if value is not None:  # Evitar salvar valores None
            setattr(associated_graph_object, attr_name, value)

    return Graph

def filter_intermediate_graphs(graphs: list, node_list):
    filtered_graphs = []

    for graph in graphs:
        filtered_graphs.append(nx.subgraph(graph, [node for node in graph.nodes() if node[0] in node_list]))
    
    return filtered_graphs  


def create_graph(edges: List[Tuple]):
    G_sub = nx.Graph()
    

    if len(edges) < 1:
        log.info(f"Edges list empty: {edges}")
        return None
    for sublist in edges:
        node_a = tuple(sublist[0]) if isinstance(sublist[0], np.ndarray) else sublist[0]
        node_b = tuple(sublist[1]) if isinstance(sublist[1], np.ndarray) else sublist[1] 
        G_sub.add_edge(node_a, node_b)
        
        chain_color_map = {}
        color_palette = plt.cm.get_cmap('tab10', 20) 
        color_counter = 1 

    # log.debug(f"Edge: {edges}, type: {type(edges)}")
    if not isinstance(edges[0][0], np.ndarray):
        for nodes in G_sub.nodes:
            chain_id = nodes[0][0]
            
            if chain_id not in chain_color_map:
                chain_color_map[chain_id] = color_palette(color_counter)[:3]  # RGB tuple
                log.debug(f"Chain_id: {chain_id}, color: {chain_color_map[chain_id]}")
                color_counter += 5


            G_sub.nodes[nodes]['chain_id'] = chain_color_map[chain_id]
    
    G_sub.remove_nodes_from(list(nx.isolates(G_sub)))
    
    return G_sub

def create_possible_nodes(reference_graph_indices: list, associated_nodes: np.ndarray, range_graph: Tuple):
    all_possible_nodes = []
    (start, end) = range_graph
    # for i in reference_graph_indices:
        
    #     elements = list(np.where(associated_nodes[:, i] > 0)[0])
        
    #     if elements:
    #         block_elements = [[i], elements]
    #         print(block_elements)
    #         all_possible_nodes.extend(list(itertools.product(*block_elements)))
    # input("Continue?")
    # return all_possible_nodes
    # log.debug(f"Reference Graph Indices: {reference_graph_indices}")
    # log.debug(f"Associated Nodes: \n{associated_nodes}")
    for i in reference_graph_indices:
        block_indices = np.where(associated_nodes[:, i] > 0)[0]

        elements = [index for index in block_indices if start <= index < end]

        if elements:

            block_elements = [[i], elements]
            # log.debug(f"{i}/{reference_graph} Making the cartesian product")
            all_possible_nodes.extend(list(itertools.product(*block_elements)))    
            # log.debug(f"{i}/{reference_graph} Cartesian product finalized")
    # log.debug(f"All possible nodes: {all_possible_nodes}")
    return all_possible_nodes
def build_contact_map(pdb_file):
    
    # Step 1: Parse the PDB file
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    
    # Step 2: Extract CA atoms
    carbon_atoms = []
    residue_map = []
    residue_map_all = []

    for model in structure:
        for chain in model:
            for residue in chain:
                
                residue_id = (chain.id, residue.id[1], residue.get_resname())
                
                residue_map.append(residue_id[0:2])
                residue_map_all.append(residue_id)
                
                
                # Checks whether the amino acid has beta carbon or not, since Glycine does not have a side chain
                carbon = "CB" if residue.has_id("CB") else "CA"
                carbon_atoms.append(residue[carbon].get_vector())

    residue_map_dict = {tup: i for i, tup in enumerate(residue_map)}
    residue_map_dict_all = {tup: i for i, tup in enumerate(residue_map_all)}

    # Step 3: Calculate distances and build contact map
    num_atoms = len(carbon_atoms)
    contact_map = np.zeros((num_atoms, num_atoms), dtype=float)
    for i in range(num_atoms-1):
        for j in range(i+1, num_atoms):
            distance = (carbon_atoms[i] - carbon_atoms[j]).norm()
            #if distance <= distance_cutoff:
            #    contact_map[i, j] = True
            #    contact_map[j, i] = True
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
    
    return atchley_factors_avg

def graph_message_passing(graph, embedding_path, use_degree, norm_features):
    '''
    This function perform a single message passing in the graph nodes to update graph node features
    It returns a dictionary with the feature nodes after message passing
    '''
    #get adjacency matrix
    adj = nx.adjacency_matrix(graph).todense()
    #get distance matrix
    pdb_df = graph.graph["pdb_df"]
    pdb_df = pdb_df.set_index('node_id', inplace=False)
    order = graph.nodes
    ordered_pdb_df = pdb_df.reindex(order)
    dist_df = compute_distmat(ordered_pdb_df)
    dist_m = dist_df.values.tolist()
    # element-wise multiplication (also known as the Hadamard product) divided by 1
    mult = 1 / (np.array(adj) * np.array(dist_m))
    mult[mult == np.inf] = 0 # replace inf to 0 
    #divide each element by the sum of the values at each row so that we end up with the weigths ranging from 0 to 1
    row_sums = np.sum(mult, axis=1)
    weights_m = mult / row_sums[:, np.newaxis]
    #multiply the weight matrix by the feature matrix
    read_emb = pd.read_csv(embedding_path)
    feature_matrix = np.array([read_emb[read_emb.AA == convert_3aa1aa(node.split(":")[1])].iloc[:,1:].values.tolist()[0] for node in graph.nodes])
    #message passing by multiplying weight matrix (former adjacency) and the feature matrix
    message_passing_m = weights_m @ feature_matrix
    #by doing the procedure above, we are updating the features of the current node only using information of the neighbor nodes, the current not informations is not incorporated, which it does not make sense for our case
    #we can either sum the Identity matrix to the adjacency matrix to include the current node features or we can concatenate the two features, current node feature and the neighbor node features
    #I'm testing the concatenation below
    concat_feat_matrix = np.concatenate((feature_matrix, message_passing_m), axis=1)
    #convert array to dict 
    node_names = list(order)
    assert len(node_names) == concat_feat_matrix.shape[0], "Number of keys must match the number of rows in the array."
    if use_degree and norm_features:
        #add degree as feature with norm
        #in this case I'm doing a normalization of 0 to 1 (considering the minimum as 1 and the maximum as 10, but the latter is not true -> need to improve this!)
        neighbors_count = {node: (len(list(graph.neighbors(node)))-1)/10 for node in graph.nodes()}
        #debug
        neighbors = {node: list(graph.neighbors(node)) for node in graph.nodes()} #-> true neighbors are computed through the whole protein not interface graph
        #norm feature matrix 0 to 1 
        concat_feat_matrix_norm = normalize_rows(concat_feat_matrix)
        feat_MP_dict = {node_names[i]: np.concatenate([concat_feat_matrix_norm[i, :], [neighbors_count[node_names[i]]]]) for i in range(concat_feat_matrix_norm.shape[0])}
    else:
        feat_MP_dict = {node_names[i]: concat_feat_matrix[i, :] for i in range(concat_feat_matrix.shape[0])} #without norm and without degree
    
    return feat_MP_dict

def cosine_similarity(array1, array2):
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
                    
                    # log.debug(f"Node {k}: {node[k]}")
                    # Obtém as coordenadas do nó e dos vizinhos
                    coord_node = get_coords_xyz(node[k], graph)
                    coord_n1 = get_coords_xyz(n1, graph)
                    coord_n2 = get_coords_xyz(n2, graph)

                    # Calcula os vetores
                    v1 = coord_n1 - coord_node
                    v2 = coord_n2 - coord_node

                    # Calcula o ângulo e armazena
                    angle = angle_between_vectors(v1, v2)
                    angles.append(angle)

                # Calcula a diferença máxima entre os ângulos de todos os pares de grafos
                for angle1, angle2 in combinations(angles, 2):
                    ang_diffs.append(abs(angle1 - angle2))

        ang_node_dict[node] = ang_diffs

    # Filtragem dos nós com todas as diferenças de ângulo abaixo de 20°
    filtered_nodes_ang = [key for key, values in ang_node_dict.items() if all(value < angle_diff for value in values)]

    return filtered_nodes_ang

