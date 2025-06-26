from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import add_distance_threshold
from graphein.protein.features.nodes.dssp import rsa, secondary_structure
from graphein.protein.config import DSSPConfig
from graphein.protein.subgraphs import extract_subgraph
from functools import partial
from time import time
import networkx as nx
import matplotlib.pyplot as plt
from utils.tools import association_product,build_contact_map, add_sphere_residues, create_subgraph_with_neighbors
import plotly.graph_objects as go
from pathlib import Path
from os import path
import os
from typing import Callable, Any, Union, Optional, List, Tuple, Dict
import numpy as np
import logging
import pandas as pd
from Bio.PDB import Structure, Model, Chain
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB import Chain, Residue, Atom
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.Superimposer import Superimposer

log = logging.getLogger("CRSProtein")


class Graph:
    """Represents a protein structure graph."""

    def __init__(self, graph_path: str, config: Optional[ProteinGraphConfig] = None):
        """
        Initialize a Graph instance.

        :param graph_path: Path to the PDB file.
        :param config: Custom configuration for the protein graph.
        """
        self.config = config or ProteinGraphConfig(
            edge_construction_functions=[partial(add_distance_threshold, long_interaction_threshold=0, threshold=10.0)],
            graph_metadata_functions=[rsa, secondary_structure],
            dssp_config=DSSPConfig(),
            granularity="centroids"
        )
        self.graph_path = graph_path
        self.graph = construct_graph(config=self.config, path=graph_path)
        self.subgraphs: Dict[str, nx.Graph] = {}
        self.depth: pd.DataFrame

    def get_subgraph(self, name:str):
        if name not in self.subgraphs.keys():
            print(f"Can't find {name} in subgraph")
            return None
        else:
            return self.subgraphs[name]
    
    def create_subgraph(self, name: str, node_list: List = [], return_node_list: bool = False, **args):
        if name in self.subgraphs.keys():
            print(f"You already have this subgraph created. Use graph.delete_subraph({name}) before creating it again.")
        elif not node_list:
            self.subgraphs[name] =  extract_subgraph(g = self.graph, **args)
            print(f"Subgraph {name} created with success!")
        elif node_list:
            self.subgraphs[name] = self.graph.subgraph(node_list)
        
        if return_node_list:
            return self.subgraphs[name].nodes
        
    def delete_subraph(self, name: str):
        if name not in  self.subgraphs.keys():
            del self.subgraphs[name]
        else:
            print(f"{name} isn't in.subgraphs")

    def filter_subgraph(self, 
            subgraph_name: str,
            filter_func: Callable[..., Any],
            name: Union[str, None] = None, 
            return_node_list: bool = False) -> Union[None, List]:
           
        nodes = [i for i in self.subgraphs[subgraph_name].nodes if filter_func(i)]
        if name:
            self.subgraphs[name] = self.subgraphs[subgraph_name].subgraph(nodes)
        else:
            self.subgraphs[subgraph_name] = self.subgraphs[subgraph_name].subgraph(nodes)
        
        if return_node_list:
            return list(self.subgraphs[name].nodes) if name is not None else list(self.subgraphs[subgraph_name].nodes)

        return None

    def join_subgraph(self, name: str, graphs_name: list, return_node_list: bool = False):
        if name in self.subgraphs.keys():
            print(f"You already have this subgraph created. Use graph.delete_subraph({name}) before creating it again.")
        elif set(graphs_name).issubset(self.subgraphs.keys()):
            
            self.subgraphs[name] = self.graph.subgraph([node for i in graphs_name for node in self.subgraphs[i].nodes])
            if return_node_list:
                return self.subgraphs[name].nodes
        else:
            print("Some of your subgraph isn't in the subgraph list")

class AssociatedGraph:
    """Handles the association of multiple protein graphs."""
    
    def __init__(self, 
                graphs: List[Tuple],  
                reference_graph: Optional[Union[str, int]] = None,
                output_path: str = ".",
                run_name: str = "",
                association_mode: str = "identity",
                association_config: Optional[Dict[str, Any]] = None):
        """
        Initialize an AssociatedGraph instance with a reduced configuration.
        
        :param graphs: List of tuples (Graph instance, raw_data) from preprocessing.
        :param reference_graph: Identifier for the reference graph.
        :param output_path: Where to save results.
        :param run_name: Unique run identifier.
        :param association_mode: "identity" or "similarity".
        :param association_config: Dictionary with keys:
                - centroid_threshold (float)
                - neighbor_similarity_cutoff (float)
                - rsa_similarity_threshold (float)
                - depth_similarity_threshold (float)
                - residues_similarity_cutoff (float)
                - angle_diff (float)
                - checks (dict)
                - factors_path (str or None)
                - residues_path (str or None)
        """
        default_config = {
            "centroid_threshold": 10.0,
            "neighbor_similarity_cutoff": 0.95,
            "rsa_similarity_threshold": 0.95,
            "depth_similarity_threshold": 0.95,
            "residues_similarity_cutoff": 0.95,
            "rsa_bins": 5.0,
            "depth_bins": 5.0,
            "distance_bins": 5.0, 
            "angle_diff": 20.0,
            "checks": {"neighbors": True, "rsa": True, "depth": True},
            "factors_path": None
        }
        self.association_config = default_config.copy()
        if association_config:
            self.association_config.update(association_config)
        
        self.graphs = graphs
        self.reference_graph = reference_graph
        self.output_path = Path(output_path)
        self.run_name = run_name
        
        self.graph_data = self._prepare_graph_data()
        
        result = association_product(graph_data=self.graph_data,
                                    association_mode=association_mode,
                                    config=self.association_config)
        
        if result is not None:
            self.__dict__.update(result)
            self.associated_graphs = result["AssociatedGraph"]
        else:
            self.associated_graphs = None

    def _prepare_graph_data(self) -> List[dict]:
        """
        For each (Graph, raw) tuple, build a dictionary with the necessary data:
            - "graph": The Graph instance.
            - "contact_map": Output of build_contact_map(raw)[0].
            - "residue_map_all": Output of build_contact_map(raw)[2].
            - "rsa": np.array(g.graph["dssp_df"]["rsa"]).
            - "depth": g.depth.
        """
        graph_data = []
        for i, (g, pdb_file) in enumerate(self.graphs):
            contact_map, residue_map, residue_map_all = build_contact_map(pdb_file)
            
            sorted_nodes = sorted(list(g.nodes()))
            depth_nodes = [str(node.split(":")[2])+node.split(":")[0] for node in sorted_nodes]
            
            data = {
                "id": i,
                "graph": g,
                "sorted_nodes": sorted_nodes,
                "depth_nodes": depth_nodes,
                "contact_map": contact_map,
                "residue_map": residue_map,
                "residue_map_all": residue_map_all,
                "rsa": g.graph["dssp_df"]["rsa"],
                "residue_depth": g.graph["depth"],
                "pdb_file": pdb_file
            }
            graph_data.append(data)
        
        return graph_data
    

    def create_pdb_per_protein(self):
        if isinstance(self.associated_graphs, list):
            for i in range(len(self.graphs)):
                pdb_file = self.graphs[i][-1]
                parser = PDBParser(QUIET=True)
                orig_struct = parser.get_structure('orig', pdb_file)
                chain_counter = 1
                new_struct = Structure.Structure('frames')
                model = Model.Model(0)
                new_struct.add(model)

                for comp_id, comps in enumerate(self.associated_graphs):
                    for frame_id in range(len(comps[0])):
                        nodes = set([node[i] for node in comps[0][frame_id].nodes])

                        chain_id = f"K{comp_id}{chain_counter:03d}"
                        chain_counter += 1
                        chain = Chain.Chain(chain_id)

                        for node in nodes:
                            parts = node.split(':')
                            chain_name = parts[0]
                            resnum = int(parts[2])

                            try:
                                orig_res = orig_struct[0][chain_name][(" ", resnum, " ")]
                            except KeyError:
                                print(f"Resíduo {node} não encontrado, pulando.")
                                continue

                            new_res = Residue.Residue(orig_res.id, orig_res.resname, orig_res.segid)

                            for atom in orig_res:
                                new_atom = Atom.Atom(
                                    atom.get_name(),
                                    atom.get_coord().copy(),
                                    atom.get_bfactor(),
                                    atom.get_occupancy(),
                                    atom.get_altloc(),
                                    atom.get_fullname(),
                                    atom.get_serial_number(),
                                    element=atom.element
                                )
                                new_res.add(new_atom)
                            # print(new_res)
                            chain.add(new_res)

                        model.add(chain)

                # Salvar a estrutura
                out_dir = Path("out")
                out_dir.mkdir(exist_ok=True)
                use_cif = True
                if use_cif:
                    io = MMCIFIO()
                    # out_file = out_dir / f"{Path(pdb_file).stem}_frames.cif"
                    out_file = f"{Path(pdb_file).stem}_frames.cif"
                else:
                    io = PDBIO()
                    # out_file = out_dir / f"{Path(pdb_file).stem}_frames.pdb"
                    out_file = f"{Path(pdb_file).stem}_frames.pdb"

                io.set_structure(new_struct)
                io.save(str(out_file))

                print(f"Estrutura salva em {out_file}")


    def _parse_label(self, label: str):
        """
        Dado 'A:GLU:154', retorna (chain_id, resnum, icode).
        Seus nós não usam insertion code, então sempre icode = ' '.
        """
        chain, _, resnum = label.split(':')
        return chain, int(resnum), ' '

    def _write_frame_multimodel(self, comp_idx: int, frame_idx: int, models: list, output_dir: str):
        """
        Cria um único mmCIF multi-model contendo todos os modelos
        (proteínas) já alinhados deste frame.
        """
        multi = Structure.Structure(f"comp{comp_idx}_frame{frame_idx}")
        for m_idx, model in enumerate(models, start=1):
            model.id = m_idx
            multi.add(model)

        os.makedirs(output_dir, exist_ok=True)
        out_path = Path(output_dir) / f"comp{comp_idx}_frame{frame_idx}_all.cif"
        io = MMCIFIO()
        io.set_structure(multi)
        io.save(str(out_path))
        print(f"[comp{comp_idx}_frame{frame_idx}] wrote {len(models)} models to {out_path}")

    def align_all_frames(self, output_dir: str):
        """
        Para cada componente (frame) em self.associated_graphs:
          - recria fresh os modelos de cada PDB em self.graphs
          - usa o modelo 0 como referência
          - alinha cada proteína móvel de acordo com as tuplas de rótulos do nó
          - salva um mmCIF multi-model com todos os modelos já alinhados
        """
        parser = PDBParser(QUIET=True)

        for comp_idx, (frame_graphs, _) in enumerate(self.associated_graphs):
            for frame_idx, assoc_graph in enumerate(frame_graphs):
                nodes = list(assoc_graph.nodes())  # tuplas de rótulos, ex ('A:ALA:4','A:ALA:6',…)

                # 1) carregar fresh cada Model para este frame
                models = []
                for prot_idx, (_, pdb_path) in enumerate(self.graphs):
                    struct = parser.get_structure(f"p{prot_idx}", pdb_path)
                    models.append(struct[0])

                # 2) extrair lista de CA da referência (prot_idx=0)
                ref_cas = []
                for label in nodes:
                    chain, resnum, icode = self._parse_label(label[0])
                    ref_res = models[0][chain][(' ', resnum, icode)]
                    ref_cas.append(ref_res['CA'])

                # 3) para cada móvel (>0), extrair CA e superimpor
                for prot_idx in range(1, len(models)):
                    mob_cas = []
                    for label in nodes:
                        chain, resnum, icode = self._parse_label(label[prot_idx])
                        mob_res = models[prot_idx][chain][(' ', resnum, icode)]
                        mob_cas.append(mob_res['CA'])

                    sup = Superimposer()
                    sup.set_atoms(ref_cas, mob_cas)
                    sup.apply(models[prot_idx].get_atoms())

                    print(
                        f"[comp{comp_idx}_frame{frame_idx}] "
                        f"prot{prot_idx} ← prot0  RMSD={sup.rms:.2f}"
                    )

                # 4) salvar todos os modelos alinhados num único mmCIF multi-model
                self._write_frame_multimodel(comp_idx, frame_idx, models, output_dir)


    def add_spheres(self):
        ...
        
  #   def draw_graph(self, show = True, save = True):
  #       if not show and not save:
  #           log.info("You are not saving or viewing the graph. Please leave at least one of the parameters as true.")
  #           return 
  #       if isinstance(self.associated_graphs, list):
  #           for i, graph in enumerate(self.associated_graphs):

  #               chain_ids = sorted({data['chain_id'] for _, data in graph.nodes(data=True)})
  #               cmap = plt.cm.get_cmap('tab10', len(chain_ids))
  #               palette = {cid: cmap(idx) for idx, cid in enumerate(chain_ids)}
  #               node_colors = [palette[graph.nodes[n]['chain_id']] for n in graph.nodes]
  #               # Draw the full cross-reactive subgraph
  #               nx.draw(graph, with_labels=True, node_color=node_colors, node_size=50, font_size=6)
  #               
  #               if show:
  #                   plt.show()
  #               if save:
  #                   if i == 0:
  #                       plt.savefig(path.join(self.output_path, "Associated Graph Base.png"))
  #                   else:

  #                       plt.savefig(path.join(self.output_path, f"Associated Graph {i}.png"))
  #                   plt.clf()
  #       
  #                   log.info(f"{i} GraphAssociated's plot saved in {self.output_path}")
  #       else:
  #           log.warning(f"I cant draw the graph because it's {self.associated_graphs}")
  # 

    def draw_graph(self, show=False, save=True):
        if not show and not save:
            log.info("You are not saving or viewing the graph. Please leave at least one of the parameters as true.")
            return

        if not isinstance(self.associated_graphs, list):
            log.warning(f"I can't draw the graph because it's {self.associated_graphs!r}")
            return

        for j, comps in enumerate(self.associated_graphs):
            for i, graph in enumerate(comps[0]): 
                # 1) Assign a chain_id to each node (hard‑coded example = "AAAA")
                print(graph)
                for node in graph.nodes():
                    # if you want to compute from the residues, do something like:
                    residues = [r for r in node]  # flatten
                    chains = [r.split(':')[0] for r in residues]
                    graph.nodes[node]['chain_id'] = ''.join(chains)

                # 2) Build color palette
                chain_ids = sorted({data['chain_id'] for _, data in graph.nodes(data=True)})
                cmap = plt.cm.get_cmap('tab10', len(chain_ids))
                palette = {cid: cmap(idx) for idx, cid in enumerate(chain_ids)}
                node_colors = [palette[graph.nodes[n]['chain_id']] for n in graph.nodes()]

                # 3) Build the labels exactly as you asked:
                node_labels = {}
                for n in graph.nodes():
                    if isinstance(n, tuple) and n and isinstance(n[0], tuple):
                        combo1, combo2 = n
                        # repr gives "('A:GLU:161', 'A:ARG:157', 'A:VAL:158')"
                        # strip spaces so you get "('A:GLU:161','A:ARG:157','A:VAL:158')"
                        lab1 = repr(combo1).replace(" ", "")
                        lab2 = repr(combo2).replace(" ", "")
                        node_labels[n] = f"{lab1}{lab2}"
                    else:
                        node_labels[n] = str(n)

                # 4) Draw it
                nx.draw(
                    graph,
                    with_labels=True,
                    labels=node_labels,
                    node_color=node_colors,
                    node_size=50,
                    font_size=6
                )

                # 5) Show/save
                if show:
                    plt.show()
                if save:
                    if i == 0 and j == 0: 
                        filename = f"Full Associated Graph Base.png"
                    elif i == 0 and j !=0: 
                        filename = f"{j} - Associated Graph Base.png"
                    else:
                        filename = f"{j} - Associated Graph {i}.png"

                    full = path.join(self.output_path, filename)
                    plt.savefig(full)
                    plt.clf()
                    log.info(f"{j}: {i}‑th associated graph saved to {full}")

    def grow_subgraph_bfs(self):
        # Build all possible common TCR interface pMHC subgraphs centered at the peptide nodes 
        count_pep_nodes = 0
        G_sub = self.associated_graph
        for nodes in G_sub.nodes:
            if nodes[0].startswith('C') and nodes[1].startswith('C'): #i.e. peptide nodes
                count_pep_nodes += 1
                #print(nodes)
                bfs_subgraph = create_subgraph_with_neighbors(self.graphs, G_sub, nodes, 20)
                for nodes2 in bfs_subgraph.nodes:
                    if nodes2[0].startswith('A') and nodes2[1].startswith('A'):
                        bfs_subgraph.nodes[nodes2]['chain_id'] = 'red'
                    elif nodes2[0].startswith('C') and nodes2[1].startswith('C'):
                        bfs_subgraph.nodes[nodes2]['chain_id'] = 'blue'
                    else:
                        bfs_subgraph.nodes[nodes2]['chain_id'] = None
                #check whether nodes meet the requirements to be a TCR interface
                number_peptide_nodes = len([i for i in bfs_subgraph.nodes if i[0].startswith('C')])
                if bfs_subgraph.number_of_nodes() >= 14 and nx.diameter(bfs_subgraph) >= 3 and number_peptide_nodes >=3:
                    node_colors = [bfs_subgraph.nodes[node]['chain_id'] for node in bfs_subgraph.nodes]
                    nx.draw(bfs_subgraph, with_labels=True, node_color=node_colors)
                    plt.savefig(path.join(self.output_path,f'plot_bfs_{nodes[0]}_{self.run_name}.png'))
                    plt.clf()

                    #write PDB with the common subgraphs as spheres
                    get_node_names = list(bfs_subgraph.nodes())
                    list_node_names = []
                    for n in range(len(self.graphs)):
                        node_names = [i[n] for i in get_node_names]
                        list_node_names.append(node_names)
  
                    #create PDB with spheres
                    add_sphere_residues(self.graphs, list_node_names, self.output_path, nodes[0])
                    #add_sphere_residues(node_names_molB, path_protein2, path_protein2.split('.pdb')[0]+'_spheres.pdb')
                else:
                    print(f"The subgraph centered at the {nodes[0]} node does not satisfies the requirements")
        if count_pep_nodes == 0:
            print(f'No peptide nodes were found in the association graph. No subgraph will be generated.')
        pass
