from __future__ import annotations

from core.config import GraphConfig, make_default_config
from core.pipeline import build_graph_with_config
from core.subgraphs import extract_subgraph
from functools import partial
from time import time
import networkx as nx
import matplotlib.pyplot as plt
from utils.tools import association_product,build_contact_map, add_sphere_residues, create_subgraph_with_neighbors
import plotly.graph_objects as go
from pathlib import Path
from os import path
import os
from typing import TypedDict, Callable, Any, Union, Optional, List, Tuple, Dict
import numpy as np
import logging
import pandas as pd
from Bio.PDB import Structure, Model, Chain
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB import Chain, Residue, Atom
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.Superimposer import Superimposer
from pyvis.network import Network
import copy


log = logging.getLogger("CRSProtein")

def _rgba_to_hex(rgba):
    r, g, b, _ = rgba
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

class GraphData(TypedDict):
    id: int
    graph: nx.Graph
    sorted_nodes: list[str]
    depth_nodes: list[str]
    contact_map: np.ndarray
    residue_map: dict
    residue_map_all: dict
    rsa: np.ndarray
    residue_depth: list[float]  
    pdb_file: str

class Graph:
    """Represents a protein structure graph (no external framework assumptions)."""

    def __init__(self, graph_path: str, config: Optional[GraphConfig] = None):
        """
        Parameters
        ----------
        graph_path : str
            Path to a PDB or mmCIF file.
        config : GraphConfig, optional
            Unified graph configuration. If not provided, a sensible default is used.
        """
        self.graph_path = graph_path
        self.config = config or make_default_config(
            centroid_threshold=8.5,
            granularity="all_atoms",  # "all_atoms" | "backbone" | "side_chain" | "ca_only"
            exclude_waters=False,
        )

        # Build the structure graph and keep only pickle-safe artifacts in G.graph
        self.graph: nx.Graph = build_graph_with_config(pdb_path=graph_path, config=self.config)

        # Convenience handles (optional; also pickle-safe)
        self.subgraphs: Dict[str, nx.Graph] = {}
        self.pdb_df: Optional[pd.DataFrame] = self.graph.graph.get("pdb_df")
        self.raw_pdb_df: Optional[pd.DataFrame] = self.graph.graph.get("raw_pdb_df")
        self.rgroup_df: Optional[pd.DataFrame] = self.graph.graph.get("rgroup_df")
        self.dssp_df: Optional[pd.DataFrame] = self.graph.graph.get("dssp_df")
        self.depth: Optional[pd.DataFrame]

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
            "rsa_bins": 5.0,
            "depth_bins": 5.0,
            "distance_bins": 5.0, 
            "angle_diff": 20.0,
            "checks": {"neighbors": True, "rsa": True, "depth": True},
            "factors_path": None,
            "exclude_waters": True
        }
        self.association_config = default_config.copy()
        if association_config:
            self.association_config.update(association_config)

        self.track_residues = self.association_config.get("track_residues", None)
        
        self.graphs = graphs
        self.reference_graph = reference_graph
        self.output_path = Path(output_path)
        self.run_name = run_name
        
        self.graph_data = self._prepare_graph_data()
        
        result = association_product(graph_data=self.graph_data,
                                    config=self.association_config)
        
        if result is not None:
            self.__dict__.update(result)
            self.associated_graphs = result["AssociatedGraph"]
        else:
            self.associated_graphs = None

    def _prepare_graph_data(self) -> List[GraphData]:
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
            contact_map, residue_map, residue_map_all = build_contact_map(pdb_file, exclude_waters=self.association_config["exclude_waters"])
          
            sorted_nodes = sorted(list(g.nodes()))
            depth_nodes = [str(node.split(":")[2])+node.split(":")[0] for node in sorted_nodes]

            data: GraphData = {
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
                            chain.add(new_res)

                        model.add(chain)

                out_dir = self.output_path / "frames"
                out_dir.mkdir(exist_ok=True)
                use_cif = True
                if use_cif:
                    io = MMCIFIO()
                    out_file = out_dir / f"{Path(pdb_file).stem}_frames.cif"
                else:
                    io = PDBIO()
                    out_file = out_dir / f"{Path(pdb_file).stem}_frames.pdb"

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

    def _write_frame_multichain(self, comp_idx: int, frame_idx: int,
                                models: list, output_dir: str):
        """
        Write one mmCIF in which *all* proteins are merged into Model 0
        and distinguished only by their (renamed) chains.

        The original chain IDs are suffixed with the protein index:
        chain A in protein 0  -> 'A0'
        chain C in protein 2  -> 'C2'
        This preserves one-letter labels when you view the file in PyMOL
        (PyMOL truncates after the first character) but keeps unique
        `_atom_site.label_asym_id` in mmCIF so nothing collides.

        mmCIF permits multi-character chain IDs, so MMCIFIO will write them
        without complaints.
        """
        combo = Structure.Structure(f"comp{comp_idx}_frame{frame_idx}")
        combo_model = Model.Model(0)
        combo.add(combo_model)

        for prot_idx, prot_model in enumerate(models):
            for ch in prot_model:
                new_chain = copy.deepcopy(ch)

                new_chain.id = f"{ch.id}{prot_idx}"
                combo_model.add(new_chain)

        os.makedirs(output_dir, exist_ok=True)
        out_path = Path(output_dir) / f"comp{comp_idx}_frame{frame_idx}_all.cif"
        io = MMCIFIO()
        io.set_structure(combo)
        io.save(str(out_path))
        print(f"[comp{comp_idx}_frame{frame_idx}] wrote {len(models)} proteins as "
            f"{len(combo_model)} chains to {out_path}")

    def align_all_frames(self):
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
                nodes = list(assoc_graph.nodes())

                models = []
                for prot_idx, (_, pdb_path) in enumerate(self.graphs):
                    struct = parser.get_structure(f"p{prot_idx}", pdb_path)
                    models.append(struct[0])

                ref_cas = []
                for label in nodes:
                    chain, resnum, icode = self._parse_label(label[0])
                    ref_res = models[0][chain][(' ', resnum, icode)]
                    ref_cas.append(ref_res['CA'])

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

                output_dir = self.output_path / "frames"
                self._write_frame_multichain(comp_idx, frame_idx, models, output_dir)


    def draw_graph_interactive(self, show=False, save=True):

        if not show and not save:
            log.info("You are not saving or viewing the graph. Please leave at least one of the parameters as true.")
            return

        if not isinstance(self.associated_graphs, list):
            log.warning(f"I can't draw the graph because it's {self.associated_graphs!r}")
            return

        out_dir = Path(self.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        for j, comps in enumerate(self.associated_graphs):
            for i, graph in enumerate(comps[0]):
                if graph.number_of_nodes() == 0:
                    log.warning(f"{j}:{i} graph has no nodes. Skipping.")
                    continue

                # chain_id per node
                for node in graph.nodes():
                    try:
                        residues = [r for r in node]  # flatten tuples
                    except TypeError:
                        residues = [str(node)]
                    chains = [str(r).split(':')[0] for r in residues]
                    graph.nodes[node]['chain_id'] = ''.join(chains) or "?"

                chain_ids = sorted({data.get('chain_id', '?') for _, data in graph.nodes(data=True)})
                cmap = plt.cm.get_cmap('tab10', max(1, len(chain_ids)))
                palette = {cid: _rgba_to_hex(cmap(idx)) for idx, cid in enumerate(chain_ids)}

                # compact labels
                node_labels = {}
                for n in graph.nodes():
                    if isinstance(n, tuple) and n and isinstance(n[0], tuple):
                        combo1, combo2 = n
                        lab1 = repr(combo1).replace(" ", "")
                        lab2 = repr(combo2).replace(" ", "")
                        node_labels[n] = f"({lab1})({lab2})"
                    else:
                        node_labels[n] = str(n)

                # attach display attrs
                for n in graph.nodes():
                    cid = graph.nodes[n].get('chain_id', '?')
                    graph.nodes[n]['label'] = node_labels[n]
                    graph.nodes[n]['title'] = f"chain: {cid}\n{node_labels[n]}"
                    graph.nodes[n]['color'] = palette.get(cid, "#999999")
                    graph.nodes[n]['size'] = 12
                    graph.nodes[n]['group'] = cid

                # relabel nodes to safe string ids
                safe_map = {n: f"v{idx}" for idx, n in enumerate(graph.nodes())}
                H = nx.relabel_nodes(graph, safe_map, copy=True)

                net = Network(
                    height="800px",
                    width="100%",
                    bgcolor="#ffffff",
                    notebook=False,
                    cdn_resources="in_line",
                    directed=graph.is_directed()
                )
                net.from_nx(H)

                # optional edge hover with weight
                for (u, v, data) in H.edges(data=True):
                    w = data.get('weight')
                    if w is not None:
                        # PyVis stores edges in net.edges, update matching one
                        # create a small title for hover
                        title = f"weight: {w}"
                        # there can be multiple edges, so update all matches
                        for e in net.edges:
                            if (e['from'] == u and e['to'] == v) or (not graph.is_directed() and e['from'] == v and e['to'] == u):
                                e['title'] = title
                                e.setdefault('value', float(w) if isinstance(w, (int, float)) else 1)

                net.set_options("""
                {
                "nodes": { "shape": "dot" },
                "interaction": { "hover": true, "tooltipDelay": 150, "zoomView": true, "dragView": true },
                "physics": {
                    "enabled": true,
                    "solver": "barnesHut",
                    "barnesHut": {
                    "gravitationalConstant": -8000,
                    "centralGravity": 0.3,
                    "springLength": 95,
                    "springConstant": 0.04,
                    "damping": 0.09,
                    "avoidOverlap": 0.1
                    },
                    "minVelocity": 0.75
                }
                }
                """)

                if save:
                    if i == 0 and j == 0:
                        filename = "Full Associated Graph Base.html"
                    elif i == 0 and j != 0:
                        filename = f"{j} - Associated Graph Base.html"
                    else:
                        filename = f"{j} - Associated Graph {i}.html"
                    full = out_dir / filename
                    html = net.generate_html(
                        notebook=False,
                        local=True
                    )

                    with open(str(full), "w+") as out:
                        out.write(html)

                    log.info(f"{j}: saved graph {i} to {full}")

                elif show:
                    tmpfile = out_dir / f"__preview_{j}_{i}.html"
                    html = net.generate_html(
                        notebook=False,
                        local=True
                    )

                    with open(str(tmpfile), "w+") as out:
                        out.write(html)

    def draw_graph(self, show=False, save=True):
        if not show and not save:
            log.info("You are not saving or viewing the graph. Please leave at least one of the parameters as true.")
            return

        if not isinstance(self.associated_graphs, list):
            log.warning(f"I can't draw the graph because it's {self.associated_graphs!r}")
            return

        for j, comps in enumerate(self.associated_graphs):
            for i, graph in enumerate(comps[0]): 
                for node in graph.nodes():
                    residues = [r for r in node]  # flatten
                    chains = [r.split(':')[0] for r in residues]
                    graph.nodes[node]['chain_id'] = ''.join(chains)

                chain_ids = sorted({data['chain_id'] for _, data in graph.nodes(data=True)})
                cmap = plt.cm.get_cmap('tab10', len(chain_ids))
                palette = {cid: cmap(idx) for idx, cid in enumerate(chain_ids)}
                node_colors = [palette[graph.nodes[n]['chain_id']] for n in graph.nodes()]

                node_labels = {}
                for n in graph.nodes():
                    if isinstance(n, tuple) and n and isinstance(n[0], tuple):
                        combo1, combo2 = n
                        lab1 = repr(combo1).replace(" ", "")
                        lab2 = repr(combo2).replace(" ", "")
                        node_labels[n] = f"{lab1}{lab2}"
                    else:
                        node_labels[n] = str(n)

                nx.draw(
                    graph,
                    with_labels=True,
                    labels=node_labels,
                    node_color=node_colors,
                    node_size=50,
                    font_size=6
                )

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
        count_pep_nodes = 0
        G_sub = self.associated_graph
        for nodes in G_sub.nodes:
            if nodes[0].startswith('C') and nodes[1].startswith('C'): #i.e. peptide nodes
                count_pep_nodes += 1

                bfs_subgraph = create_subgraph_with_neighbors(self.graphs, G_sub, nodes, 20)
                for nodes2 in bfs_subgraph.nodes:
                    if nodes2[0].startswith('A') and nodes2[1].startswith('A'):
                        bfs_subgraph.nodes[nodes2]['chain_id'] = 'red'
                    elif nodes2[0].startswith('C') and nodes2[1].startswith('C'):
                        bfs_subgraph.nodes[nodes2]['chain_id'] = 'blue'
                    else:
                        bfs_subgraph.nodes[nodes2]['chain_id'] = None

                number_peptide_nodes = len([i for i in bfs_subgraph.nodes if i[0].startswith('C')])
                if bfs_subgraph.number_of_nodes() >= 14 and nx.diameter(bfs_subgraph) >= 3 and number_peptide_nodes >=3:
                    node_colors = [bfs_subgraph.nodes[node]['chain_id'] for node in bfs_subgraph.nodes]
                    nx.draw(bfs_subgraph, with_labels=True, node_color=node_colors)
                    plt.savefig(path.join(self.output_path,f'plot_bfs_{nodes[0]}_{self.run_name}.png'))
                    plt.clf()

                    get_node_names = list(bfs_subgraph.nodes())
                    list_node_names = []
                    for n in range(len(self.graphs)):
                        node_names = [i[n] for i in get_node_names]
                        list_node_names.append(node_names)
  
                    add_sphere_residues(self.graphs, list_node_names, self.output_path, nodes[0])

                else:
                    print(f"The subgraph centered at the {nodes[0]} node does not satisfies the requirements")
        if count_pep_nodes == 0:
            print(f'No peptide nodes were found in the association graph. No subgraph will be generated.')
        pass
