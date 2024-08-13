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

def check_identity(node_pair):
    '''
    Takes a tuple of nodes in the following format: 'A:PRO:57', 'A:THR:178')
    '''
    if node_pair[0].split(':')[1] == node_pair[1].split(':')[1]:
        return True
    else:
        return False

def check_identity_same_chain(node_pair):
    '''
    Takes a tuple of nodes in the following format: 'A:PRO:57', 'A:THR:178')
    '''
    if node_pair[0].split(':')[0] == node_pair[1].split(':')[0]:
        return True
    else:
        return False

def check_cross_positions(node_pair_pair):
    '''
    Take a list of paired associated nodes like [('A:SER:71', 'A:SER:71'), ('A:GLU:161', 'A:GLU:161')]
    This function checks if a pair of association nodes have same or different positions
    Returns TRUE if in both association pairs, they are equal or both are different
    And FALSE if a pair have different positions and the other have same positions or vice-versa
    '''
    print(node_pair_pair)
    if ((node_pair_pair[0][0] == node_pair_pair[0][1]) and (node_pair_pair[1][0] == node_pair_pair[1][1])) or ((node_pair_pair[0][0] != node_pair_pair[0][1]) and (node_pair_pair[1][0] != node_pair_pair[1][1])):
        return True
    else:
        return False


def check_neighborhoods(node_pair, contact_map1, residue_map1, contact_map2, residue_map2):
    '''
    Takes a tuple of nodes in the following format: 'A:PRO:57', 'A:THR:178')
    '''

    list_neig_aa_1 = [list(residue_map1.keys())[list(residue_map1.values()).index(i)][2] for i, x in enumerate(contact_map1[residue_map1[node_pair[0].split(":")[0], int(node_pair[0].split(":")[2]), node_pair[0].split(":")[1]]]) if x < 8]
    list_neig_aa_2 = [list(residue_map2.keys())[list(residue_map2.values()).index(i)][2] for i, x in enumerate(contact_map2[residue_map2[node_pair[1].split(":")[0], int(node_pair[1].split(":")[2]), node_pair[1].split(":")[1]]]) if x < 8]

    #levenshtein requires conda install conda-forge::python-levenshtein
    lev_dist = textdistance.levenshtein.distance(list_neig_aa_1,list_neig_aa_2)
    
    if lev_dist < 5:
        return True
    else:
        return False

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



def build_contact_map(pdb_file):
    # Step 1: Parse the PDB file
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    
    # Step 2: Extract CA atoms
    ca_atoms = []
    residue_map = []
    residue_map_all = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id('CB'):
                    #print('Im not gly, have CB')
                    #residue_id = (chain.id, residue.id[1])
                    residue_id = (chain.id, residue.id[1], residue.get_resname())
                    residue_map.append(residue_id[0:2])
                    residue_map_all.append(residue_id)
                    ca_atoms.append(residue['CB'].get_vector()) #need to change the name of the object later, since it is CB not only CA 
                #elif residue.has_id('CA'):
                else:
                    #print('Im gly, not have CB')
                    residue_id = (chain.id, residue.id[1], residue.get_resname())
                    residue_map.append(residue_id[0:2])
                    residue_map_all.append(residue_id)
                    ca_atoms.append(residue['CA'].get_vector())
    residue_map_dict = {tup: i for i, tup in enumerate(residue_map)}
    residue_map_dict_all = {tup: i for i, tup in enumerate(residue_map_all)}
    # Step 3: Calculate distances and build contact map
    num_atoms = len(ca_atoms)
    contact_map = np.zeros((num_atoms, num_atoms), dtype=float)
    for i in range(num_atoms):
        for j in range(num_atoms):
            distance = (ca_atoms[i] - ca_atoms[j]).norm()
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
    #if contact_map[residue1_index, residue2_index]:
    #    return True
    #else:
    #    return False




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

def add_sphere_residues(node_names_molA, molA_path, node_names_molB, molB_path, output_path, node_name):
    # Read PDB file
    parser = PDBParser()
    structure_A = parser.get_structure('protein', molA_path)
    structure_B = parser.get_structure('protein', molB_path)

    # Create a new structure to hold the spheres
    new_structure_A = Structure.Structure("spheres")
    new_structure_B = Structure.Structure("spheres")

    # Create a new model and chain for each mol
    new_model_A = Model.Model(0)
    new_chain_A = Chain.Chain('X')
    new_model_A.add(new_chain_A)
    new_structure_A.add(new_model_A)

    new_model_B = Model.Model(0)
    new_chain_B = Chain.Chain('X')
    new_model_B.add(new_chain_B)
    new_structure_B.add(new_model_B)

    # Keep track of added residues to avoid duplicates
    added_residues_A = set()
    added_residues_B = set()

    # Add sphere residues to the new structure
    for residue_info in node_names_molA:
        chain_id, residue_name, residue_number = residue_info.split(':')
        residue_key = (chain_id, int(residue_number))
        if residue_key not in added_residues_A:
            residue = structure_A[0][chain_id][int(residue_number)]
            #ca_atom = residue['CA']
            atom_coords = [atom.coord for atom in residue]
            centroid_coords = np.mean(atom_coords, axis=0)
            sphere_residue = create_sphere_residue(residue_name, residue_number, centroid_coords)
            new_chain_A.add(sphere_residue)
            added_residues_A.add(residue_key)

    for residue_info in node_names_molB:
        chain_id, residue_name, residue_number = residue_info.split(':')
        residue_key = (chain_id, int(residue_number))
        if residue_key not in added_residues_B:
            residue = structure_B[0][chain_id][int(residue_number)]
            #ca_atom = residue['CA']
            atom_coords = [atom.coord for atom in residue]
            centroid_coords = np.mean(atom_coords, axis=0)
            sphere_residue = create_sphere_residue(residue_name, residue_number, centroid_coords)
            new_chain_B.add(sphere_residue)
            added_residues_B.add(residue_key)

    # Write the new PDB file
    io = PDBIO()
    io.set_structure(new_structure_A)
    io.save(path.join(output_path,f'spheres_molA_{node_name}.pdb'))

    io = PDBIO()
    io.set_structure(new_structure_B)
    io.save(path.join(output_path,f'spheres_molB_{node_name}.pdb'))


def create_subgraph_with_neighbors(full_graph_A, full_graph_B, association_graph, node, max_nodes):
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
        graph_matrix_A = full_graph_A.graph["pdb_df"]
        current_node_index_A = graph_matrix_A[graph_matrix_A['node_id'] == current_node[0]].index[0]
        neighbor_indices_A = [graph_matrix_A[graph_matrix_A['node_id'] == nodes[0]].index[0] for nodes in neighbors if nodes != current_node]
        dists_A = compute_distmat(graph_matrix_A).iloc[neighbor_indices_A, current_node_index_A]
        
        #Mol B
        graph_matrix_B = full_graph_B.graph["pdb_df"]
        current_node_index_B = graph_matrix_B[graph_matrix_B['node_id'] == current_node[0]].index[0]
        neighbor_indices_B = [graph_matrix_B[graph_matrix_B['node_id'] == nodes[0]].index[0] for nodes in neighbors if nodes != current_node]
        dists_B = compute_distmat(graph_matrix_B).iloc[neighbor_indices_B, current_node_index_A]

        #get a average distance for sorting
        average_list = (dists_A + dists_B)/2
        
        # Pair each nodes with its corresponding numeric value
        paired_list = list(zip(average_list, neighbors))

        # Sort the pairs based on the numeric values
        paired_list.sort()

        # Extract the sorted list of nodes
        neighbors_sorted = [string for _, string in paired_list]

        for neighbor in neighbors_sorted:
            # If neighbor is not visited and adding it won't exceed max_nodes
            #if neighbor not in visited and subgraph.number_of_nodes() + 1 <= max_nodes:
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
    cos_sim = cosine_similarity(rep_molA[node_pair[0]], rep_molB[node_pair[1]])
    #debug
    if cos_sim > threshold:
        print(node_pair[0])
        print(node_pair[1])
        print(rep_molA[node_pair[0]])
        print(rep_molB[node_pair[1]])
        print(cos_sim)

    #
    if cos_sim > threshold:
        return True
    else:
        return False
