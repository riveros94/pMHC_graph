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
from copy import deepcopy

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


def generate_edges(nodes, distance_matrix, residue_maps_unique):
    """Make edges between associated nodes using distance_matrix criteria and a filter of cross positions.
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
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node1 = nodes[i]
            node2 = nodes[j]
            
            # Checks if each residue from specific position are different
            if all(node1[k] != node2[k] for k in range(len(node1))):
                distances = [distance_matrix[node1[k], node2[k]] for k in range(len(node1))]
                
                if all(dist > 0 for dist in distances):
                    mean_distance = np.mean(distances)
                    std_distance = np.std(distances)
                    cv_distance = (std_distance / mean_distance) * 100
                    
                    if cv_distance < 15:
                        edges.append((node1, node2))
    
    edges = convert_edges_to_residues(edges, residue_maps_unique)
    
    edges = [edge for edge in edges if not set(edge[0]) == set(edge[1]) and check_cross_positions(edge)]
    
    return edges

def convert_edges_to_residues(edges: List[Tuple], residue_maps_unique: Dict) -> List[Tuple]:
    """Convert the edges that contains tuple of indices to tuple of residues

    Args:
        edges (List[Tuple]): A list that contains tuple of edges that are made of tuples of indices
        residue_maps_unique (Dict): A map that relates the indice to residue

    Returns:
        convert_edge (List[Tuple]): Return edges converted to residues notation
    """
    converted_edges = []

    for edge in edges:
        node1, node2 = edge

        converted_node1 = tuple(f"{residue_maps_unique[idx][0]}:{residue_maps_unique[idx][2]}:{residue_maps_unique[idx][1]}" for idx in node1)
        converted_node2 = tuple(f"{residue_maps_unique[idx][0]}:{residue_maps_unique[idx][2]}:{residue_maps_unique[idx][1]}" for idx in node2)

        if set(converted_node1) != set(converted_node2):
            converted_edges.append((converted_node1, converted_node2))
    
    return converted_edges

def check_multiple_neighboors(node, contact_maps, residue_maps, residue_maps_unique, ranges_graph):
    
    list_neighboors = []
    for residue_indice in node:
        for i, ranges in enumerate(ranges_graph):
            if ranges[0] <= residue_indice < ranges[1]:
                node_name = residue_maps_unique[residue_indice]
                residue_map, contact_map = residue_maps[i], contact_maps[i]
                enum = enumerate(contact_map[residue_map[node_name]])
                neighboors = [list(residue_map.keys())[list(residue_map.values()).index(i)][2] for i, x in enum if x < 8]
                list_neighboors.append(neighboors)
                break

    for list1, list2 in itertools.combinations(list_neighboors, 2):
        lev_dist = textdistance.levenshtein.distance(list1, list2)
        if lev_dist >= 5:
            return False
    else:
        return True
    
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

def filter_reduce_maps(contact_maps: List, residue_maps: List, nodes_graphs: List, distance_threshold: float = 10) -> Tuple:
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
    
    for contact_map, residue_map, nodes in zip(contact_maps, residue_maps, nodes_graphs):
        indices = [residue_map[tuple([chain, int(res_num), res_name])]
                        for chain, res_name, res_num in (node.split(":") for node in nodes)
                        if tuple([chain, int(res_num), res_name]) in residue_map]
        
        filtered_contact_map = contact_map[np.ix_(indices, indices)]
        filtered_contact_map[filtered_contact_map >= distance_threshold] = 0
        filtered_contact_maps.append(filtered_contact_map)  
    

    full_residue_maps = [
        {(chain, int(res_num), res_name): i for i, (chain, res_name, res_num) in enumerate(
            node.split(":") for node in sorted_nodes
        )}
        for sorted_nodes in nodes_graphs
    ] 
        
    return filtered_contact_maps, full_residue_maps

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
    
def create_neighboor_similarity(nodes_graphs, ranges_graph, total_lenght, neighboors, threshold=0.95):
    """Create a neighboor's similarity matrix using the cosine similiraty between the vectors that represent the neighboors from each residue.
    The comparassion is made between residues from different proteins

    Args:
        nodes_graphs (List): A list of all protein's nodes
        ranges_graph (List[Tuple]): A list of indices that indicates the position of each protein in matrix
        total_lenght (int): The total number of residues
        neighboors (dict): A dictionary that contains the residue's neighboors of each protein
        threshold (float, optional): Similarity threshold. Defaults to 0.95.

    Returns:
        matrix (np.ndarray): A numpy neigboor's similarity matrix
    """
    
    matrix = np.zeros((total_lenght, total_lenght))

    for i in range(len(nodes_graphs)):
        for j in range(i+1, len(nodes_graphs)):
            neighboorsA = neighboors[i]
            neighboorsB = neighboors[j]

            similarities = np.array([cosine_similarity(neighboorsA[product[0]], neighboorsB[product[1]]) for product in itertools.product(neighboorsA, neighboorsB)])
            similarities = np.reshape(similarities, (len(neighboorsA), len(neighboorsB)))
            
            startA, endA, startB, endB = *ranges_graph[i], *ranges_graph[j]
            
            matrix[startA:endA, startB:endB] = similarities
            matrix[startB:endB, startA:endA] = similarities.T

    matrix[matrix < threshold] = 0
    matrix[matrix >= threshold] = 1
    
    return matrix            
            
def association_product(graphs: List, association_mode: str, nodes_graphs: List, contact_maps: List, residue_maps_all: List, centroid_threshold: float = 10):
    """Make the associated graph through the cartesian product of graphs, using somem modifications to filter nodes and edges.
    Args:
        graphs (List): A list of graphs
        association_mode (str): Association mode of edges
        nodes_graphs (List): A list of nodes graphs
        contact_maps (List): A list of contact maps
        residue_maps_all (List): Full residues maps
        centroid_threshold (float, optional): Threshold to filter big distances. Defaults to 10.

    Returns:
        nx.NetworwGraph: The associated graph
    """
   
    total_lenght_graphs = sum([len(graph.nodes()) for graph in graphs])
    
    filtered_contact_maps, full_residue_maps = filter_reduce_maps(contact_maps=contact_maps, residue_maps=residue_maps_all, nodes_graphs=nodes_graphs, distance_threshold=centroid_threshold)

    prot_all_res = [[":".join(node.split(":")[:2]) for node in nodes_graph] for nodes_graph in nodes_graphs]
    prot_all_res = np.array([node for sublist in prot_all_res for node in sublist])
    
    ranges_graph = indices_graphs(graphs)
    
    neighboors_vec = {i: graph_message_passing(graph, 'resources/atchley_aa.csv', use_degree=False, norm_features=False) for i, graph in enumerate(graphs)}

    current_value = 0
    residue_maps_unique = {}
    for residue_map in full_residue_maps:
        residue_maps_unique.update({value + current_value: key for key, value in residue_map.items()})
        current_value += len(residue_map)
        

    neighboors_similarity = create_neighboor_similarity(nodes_graphs, ranges_graph, total_lenght_graphs, neighboors_vec)
    
    if association_mode == "identity":
        print("Entrei aqui")
        identity_matrix = np.equal(prot_all_res[:, np.newaxis], prot_all_res).astype(int)
        np.fill_diagonal(identity_matrix, 0)
        associated_nodes_matrix = np.multiply(neighboors_similarity, identity_matrix)
    
    elif association_mode == "similarity":

        associated_nodes_matrix = np.ones((total_lenght_graphs, total_lenght_graphs)) * neighboors_similarity
        np.fill_diagonal(associated_nodes_matrix, 0)
    
    lenght_actual = 0
    
    distance_matrix = np.zeros((total_lenght_graphs, total_lenght_graphs))    
    
    for i in range(len(graphs)):
        graph_lenght = len(graphs[i])
        
        new_lenght_actual = lenght_actual + graph_lenght
        
        associated_nodes_matrix[lenght_actual:new_lenght_actual, lenght_actual:new_lenght_actual] = 0
        distance_matrix[lenght_actual:new_lenght_actual, lenght_actual:new_lenght_actual] = filtered_contact_maps[i]
    
        lenght_actual = new_lenght_actual
        
    block_indices = {}
    all_possible_nodes = []

    reference_graph = len(graphs[0].nodes())

    for i in range(reference_graph):
        
        block_indices[i] = np.where(associated_nodes_matrix[:, i] > 0)[0]

        block_elements = [[i]]

        for start, end in ranges_graph[1:]:
            elements = [index for index in block_indices[i] if start <= index < end]
            
            if not elements:
                break
            
            block_elements.append([index for index in block_indices[i] if start <= index < end])
        else:
            all_possible_nodes.extend(list(itertools.product(*block_elements)))    

    all_possible_nodes = [node for node in all_possible_nodes if check_multiple_chains(node, residue_maps_unique)]
    
    edges = generate_edges(nodes=all_possible_nodes, distance_matrix=distance_matrix, residue_maps_unique=residue_maps_unique)

    G_sub = nx.Graph()
    
    for sublist in edges:
        node_a = sublist[0]
        node_b = sublist[1]  
        
        G_sub.add_edge(node_a, node_b)
        
    for nodes in G_sub.nodes:
        if nodes[0].startswith('A') and nodes[1].startswith('A'):
            G_sub.nodes[nodes]['chain_id'] = 'red'
        elif nodes[0].startswith('C') and nodes[1].startswith('C'):
            G_sub.nodes[nodes]['chain_id'] = 'blue'
        else:
            G_sub.nodes[nodes]['chain_id'] = None
    
    G_sub.remove_nodes_from(list(nx.isolates(G_sub)))
    print(f"G_sub: {G_sub}")
    return G_sub

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

'''
def add_sphere_residues_old(node_names_molA, molA_path, node_names_molB, molB_path, output_path, ref_model):
    # Read PDB file
    parser = PDBParser()
    structure_A = parser.get_structure('protein', inputmolA_path_pdb)
    structure_B = parser.get_structure('protein', inputmolA_path_pdb)

    # Create a new structure to hold the spheres
    new_structure = Structure.Structure("spheres")

    # Create a new model and chain
    new_model = Model.Model(0)
    new_chain = Chain.Chain('X')
    new_model.add(new_chain)
    new_structure.add(new_model)

    # Keep track of added residues to avoid duplicates
    added_residues = set()

    # Add sphere residues to the new structure
    for residue_info in residues_list:
        chain_id, residue_name, residue_number = residue_info.split(':')
        residue_key = (chain_id, int(residue_number))
        if residue_key not in added_residues:
            residue = structure[0][chain_id][int(residue_number)]
            ca_atom = residue['CA']
            sphere_residue = create_sphere_residue(residue_name, residue_number, ca_atom.get_coord())
            new_chain.add(sphere_residue)
            added_residues.add(residue_key)

    # Write the new PDB file
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb)
'''

# def add_sphere_residues(node_names_molA, molA_path, node_names_molB, molB_path, output_path, node_name):
#     # Read PDB file
#     parser = PDBParser()
#     structure_A = parser.get_structure('protein', molA_path)
#     structure_B = parser.get_structure('protein', molB_path)

#     # Create a new structure to hold the spheres
#     new_structure_A = Structure.Structure("spheres")
#     new_structure_B = Structure.Structure("spheres")

#     # Create a new model and chain for each mol
#     new_model_A = Model.Model(0)
#     new_chain_A = Chain.Chain('X')
#     new_model_A.add(new_chain_A)
#     new_structure_A.add(new_model_A)

#     new_model_B = Model.Model(0)
#     new_chain_B = Chain.Chain('X')
#     new_model_B.add(new_chain_B)
#     new_structure_B.add(new_model_B)

#     # Keep track of added residues to avoid duplicates
#     added_residues_A = set()
#     added_residues_B = set()

#     # Add sphere residues to the new structure
#     for residue_info in node_names_molA:
#         chain_id, residue_name, residue_number = residue_info.split(':')
#         residue_key = (chain_id, int(residue_number))
#         if residue_key not in added_residues_A:
#             residue = structure_A[0][chain_id][int(residue_number)]
#             #ca_atom = residue['CA']
#             atom_coords = [atom.coord for atom in residue]
#             centroid_coords = np.mean(atom_coords, axis=0)
#             sphere_residue = create_sphere_residue(residue_name, residue_number, centroid_coords)
#             new_chain_A.add(sphere_residue)
#             added_residues_A.add(residue_key)

#     for residue_info in node_names_molB:
#         chain_id, residue_name, residue_number = residue_info.split(':')
#         residue_key = (chain_id, int(residue_number))
#         if residue_key not in added_residues_B:
#             residue = structure_B[0][chain_id][int(residue_number)]
#             #ca_atom = residue['CA']
#             atom_coords = [atom.coord for atom in residue]
#             centroid_coords = np.mean(atom_coords, axis=0)
#             sphere_residue = create_sphere_residue(residue_name, residue_number, centroid_coords)
#             new_chain_B.add(sphere_residue)
#             added_residues_B.add(residue_key)

#     # Write the new PDB file
#     io = PDBIO()
#     io.set_structure(new_structure_A)
#     io.save(path.join(output_path,f'spheres_molA_{node_name}.pdb'))

#     io = PDBIO()
#     io.set_structure(new_structure_B)
#     io.save(path.join(output_path,f'spheres_molB_{node_name}.pdb'))

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
    duplicated_node_ids = pdb_df[pdb_df.duplicated(subset='node_id', keep=False)]

    # Exiba as labels duplicadas
    if not duplicated_node_ids.empty:
        print("Duplicated node_ids found:")
        print(duplicated_node_ids['node_id'].values)
    else:
        print("No duplicated node_ids found.")
    pdb_df = pdb_df.set_index('node_id', inplace=False)

    order = []
    for node in graph.nodes:
        if isinstance(node, tuple):
            node_id = "|".join(node)
        else:
            node_id = node
        order.append(node_id)
    ordered_pdb_df = pdb_df.reindex(order)
    dist_df = compute_distmat(ordered_pdb_df)
    dist_m = dist_df.values.tolist()

    mult = 1 / (np.array(adj) * np.array(dist_m))
    mult[mult == np.inf] = 0 # replace inf to 0 
    row_sums = np.sum(mult, axis=1)
    weights_m = mult / row_sums[:, np.newaxis]

    #multiply the weight matrix by the feature matrix
    read_emb = pd.read_csv(embedding_path)
    feature_matrix = np.array([calculate_atchley_average(node, read_emb) for node in graph.nodes]) 

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
