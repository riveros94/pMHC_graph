from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import add_distance_threshold
from functools import partial
import pandas as pd
from pathlib import Path
import os
from os import path
import argparse
from graph.graph import *
from config.graph_config import make_graph_config
from io_utils.pdb_io import list_pdb_files, get_user_selection
import logging, sys
import json
import numpy
import matplotlib
from typing import Dict, List, Optional
from SERD import read_vdw, read_pdb, get_vertices, surface, interface, _get_sincos


class Surface(object):

    def __init__(
        self, grid: numpy.ndarray, step: float, probe: float, vertices: numpy.ndarray
    ):
        self.grid = grid
        self.step = step
        self.probe = probe
        self.vertices = vertices
        self.coordinates = self._get_coordinates(grid, step, vertices)

    def _get_coordinates(
        self, grid: numpy.ndarray, step: float, vertices: numpy.ndarray
    ) -> numpy.ndarray:
        """
        Convert the grid representation of the surface to 3D Cartesian coordinates.

        Parameters
        ----------
        grid : numpy.ndarray
            The grid representation of the surface.
        step : float
            The step size used to model the surface.
        vertices : numpy.ndarray
            The vertices of the bounding box. P1: origin, P2: x-axis, P3: y-axis, P4: z-axis.

        Returns
        -------
        numpy.ndarray
            The 3D Cartesian coordinates of the surface.
        """
        indexes = numpy.argwhere(grid == 1)

        # P1, P2, P3, P4 (origin, x-axis, y-axis, z-axis)
        P1, _, _, _ = vertices

        # Calculate sin and cos for each axis
        sincos = _get_sincos(vertices)

        # Convert grid to 3D Cartesian coordinates
        xaux, yaux, zaux = (indexes * step).T

        x = (
            (xaux * sincos[3])
            + (yaux * sincos[0] * sincos[2])
            - (zaux * sincos[1] * sincos[2])
            + P1[0]
        )
        y = (yaux * sincos[1]) + (zaux * sincos[0]) + P1[1]
        z = (
            (xaux * sincos[2])
            - (yaux * sincos[0] * sincos[3])
            + (zaux * sincos[1] * sincos[3])
            + P1[2]
        )

        # Prepare 3D coordinates
        coordinates = numpy.array([x, y, z]).T

        return coordinates


class Structure(object):

    def __init__(self, vdw: Optional[str] = None, **kwargs):
        self.__dict__.update(kwargs)
        self.vdw = read_vdw(vdw)
        self.atomic = None
        self.surface = None

    def load(self, path: str):
        """
        Load the atomic data from a PDB file.

        Parameters
        ----------
        path : str
            The path to the PDB file.
        """
        self.atomic = read_pdb(path)

    def model_surface(self, type: str = "SES", step: float = 0.6, probe: float = 1.4):
        """
        Model the surface of the structure using the atomic data.
        The surface is modeled using the Solvent Excluded Surface (SES) or Solvent Accessible Surface (SAS) method. The SES method is used by default.

        Parameters
        ----------
        type : str, optional
            The type of surface to model, either 'SES' or 'SAS', by default 'SES'.
            SES: Solvent Excluded Surface. SAS: Solvent Accessible Surface.
        step : float, optional
            The step size used to model the surface, by default 0.6.
        probe : float, optional
            The radius of the probe used to model the surface, by default 1.4.

        Raises
        ------
        ValueError
            If no atomic data is loaded, raise an error.
        """
        if self.atomic is None:
            raise ValueError("No atomic data loaded. Please run .load() first.")

        # Calculate vertices of the bounding box
        vertices = get_vertices(self.atomic)

        # Model surface representation
        _surface = surface(
            self.atomic, surface_representation=type, step=step, probe=probe
        )
        self.surface = Surface(_surface, step, probe, vertices)

    def _get_interface(self, ignore_backbone: bool = True) -> Dict[str, List[int]]:
        return interface(
            self.surface.grid,
            self.atomic,
            ignore_backbone=ignore_backbone,
            step=self.surface.step,
            probe=self.surface.probe,
        )

    def atom_depth(self) -> pd.DataFrame:
        """
        Calculate the depth of each atom in the structure. The atom radius is subtracted from the minimum distance to the surface.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the depth of each atom in the structure.
        """
        if surface is None:
            raise ValueError(
                "No surface data loaded. Please run .model_surface() first."
            )

        # Get coordinates from atomic
        atomic_coordinates = self.atomic[:, 4:7].astype(float)

        # Calculate distances between surface and atomic coordinates
        distances = numpy.sqrt(
            (
                (
                    self.surface.coordinates[:, numpy.newaxis, :]
                    - atomic_coordinates[numpy.newaxis, :, :]
                )
                ** 2
            ).sum(axis=2)
        )

        # Get minimum distance for each atom
        atom_depth = distances.min(axis=0) - self.atomic[:, 7].astype(float)

        # Prepare data
        data = pd.DataFrame(
            self.atomic[:, 0:4],
            columns=["ResidueNumber", "Chain", "ResidueName", "AtomName"],
        )
        data["AtomicDepth"] = atom_depth

        return data

    def residue_depth(
        self,
        metric: str = "minimum",
        keep_only_interface: bool = False,
        ignore_backbone: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate the depth of each residue in the structure. The residue depth is calculated as the minimum or centroid of the atoms in the residue.

        Parameters
        ----------
        metric : str, optional
            The metric used to calculate the residue depth, either 'minimum', 'centroid', by default 'minimum'.
        keep_only_interface : bool, optional
            Whether to keep only residues at the interface, by default False.
        ignore_backbone : bool, optional
            Whether to ignore backbone atoms for defining the interface, by default True.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the depth of each residue in the structure.
        """
        # Calculate atom depth
        atom_depth = self.atom_depth()

        # Keep only residues at the interface
        if keep_only_interface:
            interface_residues = pd.DataFrame(
                self._get_interface(ignore_backbone=ignore_backbone),
                columns=["ResidueNumber", "Chain", "ResidueName"],
            )

            # Keep only the interface
            atom_depth = pd.merge(
                atom_depth,
                interface_residues,
                on=["ResidueNumber", "Chain", "ResidueName"],
                how="inner",
            )

        # Calculate residue depth
        if metric == "minimum":
            residue_depth = (
                atom_depth.groupby(["ResidueNumber", "Chain"], sort=False)
                .agg({"AtomicDepth": "min"})
                .reset_index()
            )
        elif metric == "centroid":
            residue_depth = (
                atom_depth.groupby(["ResidueNumber", "Chain"], sort=False)
                .agg({"AtomicDepth": "mean"})
                .reset_index()
            )
        else:
            raise ValueError("Invalid metric. Please use 'minimum' or 'centroid'.")

        # Rename column for consistency
        residue_depth.rename(columns={"AtomicDepth": "ResidueDepth"}, inplace=True)

        return residue_depth

def select_residues_within_range(structure: Structure, range_size=3):
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
    residue_depth = structure.residue_depth()

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
    atomic_data = structure.atomic
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
'''
#This script takes two pMHC structures as input and returns the common subgraphs, which will also be mapped to the structure

#The input PDB structures must be unbound, without the TCR, only the pMHC structure

#original work of a similar algorithm can be found in Protein−Protein Binding-Sites Prediction by Protein Surface Structure Conservation Janez Konc and Dušanka Janežič, 2006

#Graphein was tested inside a conda enviroment

#Graphein was not computing correctly the centoids when multiple chains were included, I correction I made was replace
#["residue_number", "chain_id", "residue_name", "insertion"] by
#["chain_id", "residue_number", "residue_name", "insertion"] in anaconda3/envs/graphein/lib/python3.8/site-packages/graphein/protein/graphs.py

#I found some incompabilities when running dssp with graphein, so I had to made a correction in my graphein version:
#In anaconda3/envs/graphein/lib/python3.8/site-packages/graphein/protein/features/nodes/dssp.py
#I changed #pdb_file, DSSP=executable, dssp_version=dssp_version to 
#pdb_file, DSSP=executable
'''



def none_or_float(value):
    if value == 'None':
        return None
    return float(value)

#command example
#python3 --molA_path /home/helder/Projects/pMHC_graphs/pdbs_teste/pmhc_mage3_5brz_renumber.pdb --molB_path /home/helder/Projects/pMHC_graphs/pdbs_teste/pmhc_titin_5bs0_renumber.pdb --interface_list /home/helder/Projects/pMHC_graphs/interface_MHC_unique.csv --centroid_threshold 10 --run_name teste_sim --association_mode similarity --output_path output5

### functions

def parser_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Building common subgraphs')
    parser.add_argument('--mols_path', type=str, default='',
                        help='Path with PDB input files.')
    # parser.add_argument('--interface_list', type=str, default='',
                        # help='File with a canonical list of MHC residues at the interface with TCR. No header needed for this file.')
    parser.add_argument('--centroid_threshold', type=int, default=10,
                        help="Distance threshold for building the molA and molB interface graphs")
    parser.add_argument('--run_name', type=str, default='test',
                        help='Name for storing results in the output folder')
    parser.add_argument('--association_mode', type=str, default='identity',
                        help='Mode for creating association nodes. Identify or similarity.')                                        
    parser.add_argument('--output_path', type=str, default='~/',
                        help='Path to store output results.')
    parser.add_argument('--neighbor_similarity_cutoff', type=float, default=0.95,
                        help="Threshold for neighbor's similarity ")
    parser.add_argument('--residues_similarity_cutoff', type=float, default=0.95,
                        help="Threshold for residue's similarity")
    parser.add_argument('--factors_path', type=str, default=None,
                        help="Factors for calculating the residue similarity ")
    parser.add_argument('--rsa_filter', type=none_or_float, default=0.1,
                        help="Threshold for filter residues by RSA")
    parser.add_argument('--rsa_similarity_threshold', type=float, default=0.90,
                        help="Threshold for make an associate graph using RSA similarity")
    parser.add_argument('--residues_lists', type=str, default=None,
                        help="Path to Json file which contains the pdb residues")
    parser.add_argument('--debug', type=bool, default=False,
                        help="Activate debug mode")

    parser.add_argument(
        "--vdw", type=str, default=None, help="Path to VDW file (optional)"
    )
    parser.add_argument(
        "--step", type=float, default=0.6, help="Step size for surface modeling"
    )
    parser.add_argument(
        "--probe", type=float, default=1.4, help="Probe radius for surface modeling"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="SES",
        choices=["SES", "SAS"],
        help="Type of surface to model (SES or SAS)",
    )
    parser.add_argument(
        "--keep_only_interface",
        action="store_true",
        help="Keep only residues at the interface",
    )
    parser.add_argument(
        "--ignore_backbone",
        action="store_true",
        help="Ignore backbone atoms for defining the interface",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="minimum",
        choices=["minimum", "centroid"],
        help="Metric used to calculate the residue depth",
    )
    args = parser.parse_args()
    
    return args

def get_exposed_residues(graph: Graph, rsa_filter = 0.1, params=None):
    all_residues = []
    merge_graphs = []
    
    if rsa_filter is None:
        graph.create_subgraph(name="exposed_residues")
    else:
        graph.create_subgraph(name="exposed_residues", rsa_threshold = rsa_filter)
        
    if "chains" not in params.keys() and "residues" not in params.keys():

        return graph.get_subgraph(name="exposed_residues")

    if "chains" in params.keys():
        if not isinstance(params["chains"], list):
            print(f"`chains` key must have a list. I received: {params['chains']}, {type(params['chains'])}")
            exit(0)
        
        graph.create_subgraph(name= "selected_chains", chains=params["chains"])
        merge_graphs.append("selected_chains")
        
    if "residues" in params.keys():
        if not isinstance(params["residues"], dict):
            print(f"`dict` key must have a list. I received: {params['dict']}, {type(params['dict'])}")
            exit(0)
        
        residues_dict = params["residues"]
        
        for chain in residues_dict:
            if not isinstance(residues_dict[chain], list):
                print(f"The key {chain} in residues isn't a list")
                exit(0)
            
            all_residues += residues_dict[chain]
            
        graph.create_subgraph(name="all_residues", sequence_positions=all_residues)
        merge_graphs.append("all_residues")
    
    graph.join_subgraph(name="merge_list", graphs_name=merge_graphs)
    
    exposed_nodes = graph.filter_subgraph(subgraph_name="exposed_residues", name="selected_exposed_residues", filter_func= lambda i: i in graph.subgraphs["merge_list"], return_node_list=True)
    graph.create_subgraph(name = "s_graph", node_list=exposed_nodes, return_node_list=False)
    
    return graph.get_subgraph("s_graph")

def get_exposed_residues_mhc(graph: Graph, inter_list, rsa_filter = 0.1, chains_peptide = ["C"], chain_mhc = "A"):
    graph.create_subgraph(name= "peptide_list", chains=chains_peptide)
    graph.create_subgraph(name="mhc_list", sequence_positions=inter_list)
    graph.filter_subgraph(subgraph_name="mhc_list", filter_func= lambda i: i[0] == chain_mhc)
    graph.create_subgraph(name="all_solv_exposed", rsa_threshold = rsa_filter)
    graph.join_subgraph(name="all_list", graphs_name=["peptide_list", "mhc_list"])
    lista_peptide_helix_exposed = graph.filter_subgraph(subgraph_name="all_solv_exposed", name="peptide_helix_exposed_list", filter_func = lambda i: i in graph.subgraphs["all_list"], return_node_list=True)
    graph.create_subgraph(name = "s_graph", node_list=lista_peptide_helix_exposed, return_node_list= False)

    return graph.get_subgraph("s_graph")

def main():
    args = parser_args()

    #### Inputs 

    # List of MHC interface residues with TCR from the non-redundant TCR:pMHC analysis
    # inter_list = pd.read_csv(args.interface_list, header=None)[0].to_list()
    
    centroid_threshold=args.centroid_threshold
    rsa_filter = args.rsa_filter
    rsa_similarity_threshold = args.rsa_similarity_threshold
    neighbor_similarity_cutoff = args.neighbor_similarity_cutoff
    
    debug = args.debug
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    if debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        log = logging.getLogger("CRSProtein") 
        log.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        log = logging.getLogger("CRSProtein") 
        log.setLevel(logging.INFO)
    
    with open(args.residues_lists, "r") as f:
        residues_lists = json.load(f) 


    # List of paths
    mols_path = args.mols_path
    mols_files = list_pdb_files(mols_path)
    if not mols_files:
        return
        
    selected_files, reference_graph = get_user_selection(mols_files, mols_path)

    output_path = args.output_path
    #Path to full common subgraph
    path_full_subgraph = path.join(output_path,f"full_association_graph_{args.run_name}.png")
    ################################

    #check if output folder exists, otherwise create it 
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Initialize the protein graph config
    config = make_graph_config(centroid_threshold=centroid_threshold)
    
    graphs = []

    for mol_path in selected_files:
        g = Graph(config=config, graph_path=mol_path[0])

        structure = Structure(vdw=args.vdw)
        structure.load(mol_path[0])
        structure.model_surface(type=args.type, step=args.step, probe=args.probe)
        # Seleciona os resíduos com numeração próxima
        # selected_residues = select_residues_within_range(structure)

        # Adiciona esferas aos resíduos selecionados e salva o PDB
        # add_spheres_to_pdb(structure, selected_residues, output_pdb="output_with_spheres.pdb")

        # print("Estrutura com esferas foi salva em 'output_with_spheres.pdb'.")

        # input("Continuar?")
        params = residues_lists[mol_path[1]]
        
        if "base" in params.keys():
            base = params["base"]
            try:
                params = residues_lists[base]
            except KeyError as e:
                raise KeyError(f"I wasn't able to find the template for {base}. Error message: {e}")            
    
        # s_g = get_exposed_residues_mhc(g, inter_list=inter_list, rsa_filter = rsa_filter, chains_peptide=["C"], chain_mhc="A")
        s_g = get_exposed_residues(graph=g, rsa_filter=rsa_filter, params=params )
        s_g.residue_depth = structure.residue_depth(metric=args.metric, keep_only_interface=args.keep_only_interface, ignore_backbone=args.ignore_backbone)  
        
        # logging.info(f"Residue Depth: {s_g.residue_depth}")
        graphs.append((s_g, mol_path[0]))

    G = AssociatedGraph(graphs=graphs, reference_graph= reference_graph, output_path=output_path, path_full_subgraph=path_full_subgraph, association_mode=args.association_mode, factors_path=args.factors_path, run_name= args.run_name, centroid_threshold=centroid_threshold, residues_similarity_cutoff=args.residues_similarity_cutoff, neighbor_similarity_cutoff=neighbor_similarity_cutoff, rsa_similarity_threshold=rsa_similarity_threshold)
    # G_sub = G.associated_graph

    G.draw_graph(show = True)
    # print(G.associated_graph.nodes())

    G.grow_subgraph_bfs()

if __name__ == "__main__":

    # Verificar o número de threads OpenMP
    num_threads = os.environ.get('OMP_NUM_THREADS', None)
    if num_threads:
        print(f"Usando {num_threads} threads OpenMP.")
    else:
        print("A variável OMP_NUM_THREADS não está definida.")

    main()
    