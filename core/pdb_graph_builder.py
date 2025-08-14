
"""
pdb_graph_builder.py
---------------------
Gerador de grafos estruturais de pMHC usando Bio.PDB + NetworkX.

Funcionalidades principais:
  • Parse com Bio.PDB
  • Cálculo de centróides por resíduo e por molécula de água
  • Cálculo de ASA via Shrake-Rupley e conversão para RSA usando Tien et al. (2013)
  • Construção de grafo com NetworkX a partir de um cutoff de distância
  • Opção de incluir nós de água e arestas resíduo-água

Requisitos:
  biopython >= 1.79  (Bio.PDB)
  networkx
  numpy, scipy

Autor: IC Helper
"""
from __future__ import annotations

import os
import re
import networkx as nx
import numpy as np
import pandas as pd

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Literal


from Bio.PDB import PDBParser, MMCIFParser, ShrakeRupley
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.DSSP import DSSP, dssp_dict_from_pdb_file, residue_max_acc

log = logging.getLogger(__name__)


AA1_TO_3 = {
    "A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","Q":"GLN","E":"GLU","G":"GLY",
    "H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE","P":"PRO","S":"SER",
    "T":"THR","W":"TRP","Y":"TYR","V":"VAL"
}

BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT"}

def _is_protein_residue(res: Residue) -> bool:
    """Resíduo proteico padrão (exclui HETATM, água, íons)."""
    hetflag, _, _ = res.id
    return hetflag == " " and is_aa(res, standard=True)


def _is_water(res: Residue) -> bool:
    """Identifica moléculas de água representadas como resíduos HOH."""
    return res.get_resname() == "HOH"


def _heavy_atom_coords(res: Residue) -> np.ndarray:
    """Retorna as coordenadas dos átomos não hidrogênio do resíduo."""
    coords: List[np.ndarray] = []
    for atom in res.get_atoms():
        # Alguns PDBs não têm elemento anotado, usar id do átomo como fallback
        element = getattr(atom, "element", None)
        if element is None:
            name = atom.get_name().upper()
            if name.startswith("H"):
                continue
        else:
            if element.upper() == "H":
                continue
        coords.append(atom.coord)  # type: ignore[attr-defined]
    if not coords:
        # Fallback para qualquer átomo se não encontrar heavy, evita NaN
        coords = [atom.coord for atom in res.get_atoms()]  # type: ignore[attr-defined]
    return np.asarray(coords, dtype=float)


def _centroid(coords: np.ndarray) -> np.ndarray:
    """Centroid de um conjunto de coordenadas Nx3."""
    return coords.mean(axis=0)


def _node_id(chain_id: str, res: Residue, kind: str = "residue") -> str:
    """Gera um identificador único para nó no grafo."""
    hetflag, resseq, icode = res.id
    resname = res.get_resname()
    if kind == "water":
        return f"{chain_id}:HOH:{resseq}{icode.strip() or ''}"
    return f"{chain_id}:{resname}:{resseq}{icode.strip() or ''}"


@dataclass
class GraphBuildConfig:
    chains: Optional[Iterable[str]] = None            # se None, usa todas
    include_waters: bool = False                      # incluir moléculas de água
    residue_distance_cutoff: float = 10.0             # cutoff resíduo-resíduo em Å
    water_distance_cutoff: float = 6.0                # cutoff água-resíduo em Å
    store_distance_matrix: bool = False               # armazenar a matriz de distâncias no objeto
    compute_rsa: bool = True                          # calcular ASA e RSA
    probe_radius: float = 1.4                         # raio do solvente para Shrake-Rupley
    n_points: int = 960                               # pontos por átomo para SR
    model_index: int = 0                              # qual modelo utilizar do PDB
    allow_empty_chains: bool = False                  # não falhar se cadeia listada não existir
    rsa_method: Literal["sr", "dssp"] = "dssp"                  # "sr" = Shrake–Rupley, "dssp" = igual Graphein
    dssp_exec: str = "mkdssp"                                 
    dssp_acc_array: Literal["Sander", "Wilke", "Miller"] = "Sander"  # tabelas internas do Bio.PDB.DSSP

@dataclass
class BuiltGraph:
    graph: nx.Graph
    residue_index: List[Tuple[str, Residue]]                  # [(node_id, residue)]
    residue_centroids: np.ndarray                             # shape (N, 3)
    water_index: List[Tuple[str, Residue]] = field(default_factory=list)
    water_centroids: Optional[np.ndarray] = None
    distance_matrix: Optional[np.ndarray] = None
    raw_pdb_df: Optional[pd.DataFrame] = None
    pdb_df: Optional[pd.DataFrame] = None
    rgroup_df: Optional[pd.DataFrame] = None


class PDBGraphBuilder:
    """
    Constrói grafos estruturais a partir de arquivos PDB ou mmCIF.

    Uso típico:
        cfg = GraphBuildConfig(chains=["A","C"], include_waters=True)
        builder = PDBGraphBuilder("path/to/file.pdb", cfg)
        built = builder.build_graph()
        G = built.graph
    """

    def __init__(self, pdb_path: str, config: Optional[GraphBuildConfig] = None) -> None:
        self.pdb_path = pdb_path
        self.config = config or GraphBuildConfig()
        self.structure: Optional[Structure] = None

    # -----------------------
    # Carregamento da estrutura
    # -----------------------
    def load(self) -> None:
        """Carrega a estrutura a partir de um PDB ou mmCIF."""
        if self.pdb_path.lower().endswith(".cif") or self.pdb_path.lower().endswith(".mmcif"):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        self.structure = parser.get_structure("struct", self.pdb_path)
        log.info("Estrutura carregada de %s", self.pdb_path)

    # -----------------------
    # ASA e RSA
    # -----------------------
    def _compute_asa_rsa(
        self, res_tuples: List[Tuple[str, Residue, np.ndarray]]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Retorna {node_id: (ASA_absoluto, RSA_relativo)}.
        - 'dssp': usa RSA do DSSP (idx 3) e ASA = RSA * residue_max_acc[acc_array][AA3]
        - 'sr'  : Shrake–Rupley e normaliza com residue_max_acc[acc_array][AA3]
        """
        # Tabela interna do BioPython para máximos ASA
        max_acc_table = residue_max_acc[self.config.dssp_acc_array]  # ex.: "Sander"

        if self.config.rsa_method == "dssp":
            assert self.structure is not None
            model = self.structure[self.config.model_index]
            dssp = DSSP(model, self.pdb_path, dssp=self.config.dssp_exec,
                        acc_array=self.config.dssp_acc_array)

            # mapeia (chain, (' ', resseq, icode)) -> node_id
            idx2nid = {(res.get_parent().id, res.id): nid for nid, res, _ in res_tuples}

            out: Dict[str, Tuple[float, float]] = {}
            for key in dssp.keys():
                nid = idx2nid.get(key)
                if nid is None:
                    continue
                tup = dssp[key]
                aa1 = tup[1]            # 1 letra
                rsa = float(tup[3])     # relativo (já normalizado pela acc_array escolhida)
                aa3 = AA1_TO_3.get(aa1, "UNK")
                max_acc = float(max_acc_table.get(aa3, 0.0))
                asa = rsa * max_acc if max_acc > 0 else 0.0
                out[nid] = (asa, rsa)
            return out

        # ---- modo SR (Shrake–Rupley) com normalização pela mesma tabela do BioPython ----
        assert self.structure is not None
        sr = ShrakeRupley(probe_radius=self.config.probe_radius, n_points=self.config.n_points)
        sr.compute(self.structure, level="R")

        out: Dict[str, Tuple[float, float]] = {}
        for nid, res, _ in res_tuples:
            asa = float(getattr(res, "sasa", 0.0))
            aa3 = res.get_resname()
            max_acc = float(max_acc_table.get(aa3, 0.0))
            rsa = (asa / max_acc) if max_acc > 0 else 0.0
            out[nid] = (asa, rsa)
        return out
    
    def _make_atom_tables(self, chains: List[Chain]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Gera raw_pdb_df (tudo), pdb_df (proteína heavy atoms), rgroup_df (side chain heavy).
        Colunas: record_name, atom_number, atom_name, alt_loc, residue_name, chain_id,
                residue_number, insertion, x_coord, y_coord, z_coord, occupancy,
                b_factor, element_symbol, charge, model_idx, node_id, residue_id
        """
        rows = []
        model_idx = self.config.model_index
        for ch in chains:
            for res in ch.get_residues():
                hetflag, resseq, icode = res.id
                resname = res.get_resname()
                chain_id = ch.id
                is_water = (resname == "HOH")
                is_std = _is_protein_residue(res)

                # record_name: ATOM para proteína padrão; HETATM para água/ligantes/outros
                record_name = "ATOM" if is_std else "HETATM"
                node_id = _node_id(chain_id, res, kind="water" if is_water else "residue")
                residue_id = f"{chain_id}:{int(resseq)}{(icode.strip() or '')}"

                for atom in res.get_atoms():
                    try:
                        serial = atom.get_serial_number()
                    except Exception:
                        serial = None
                    name = atom.get_name().strip()
                    try:
                        alt = atom.get_altloc()
                    except Exception:
                        alt = ""
                    element = getattr(atom, "element", None)
                    if not element:
                        element = name[0] if name else ""
                    x, y, z = map(float, atom.coord)  # type: ignore[attr-defined]
                    occ = atom.get_occupancy()
                    bfac = atom.get_bfactor()
                    charge = None  # Bio.PDB geralmente não mantém carga formal

                    rows.append({
                        "record_name": record_name,
                        "atom_number": serial,
                        "atom_name": name,
                        "alt_loc": (alt or "").strip(),
                        "residue_name": resname,
                        "chain_id": chain_id,
                        "residue_number": int(resseq),
                        "insertion": (icode.strip() or ""),
                        "x_coord": x, "y_coord": y, "z_coord": z,
                        "occupancy": occ, "b_factor": bfac,
                        "element_symbol": (str(element).upper() if element else ""),
                        "charge": charge,
                        "model_idx": model_idx,
                        "node_id": node_id,
                        "residue_id": residue_id,
                    })

        raw_df = pd.DataFrame(rows)

        # pdb_df: proteína padrão, heavy atoms, sem água
        pdb_df = raw_df[
            (raw_df["record_name"] == "ATOM") &
            (raw_df["element_symbol"] != "H") &
            (raw_df["residue_name"] != "HOH")
        ].copy()

        # rgroup_df: remove backbone (N, CA, C, O, OXT)
        rgroup_df = pdb_df[~pdb_df["atom_name"].str.upper().isin(BACKBONE_NAMES)].copy()

        # reordenar colunas conforme solicitado
        cols = [
            "record_name","atom_number","atom_name","alt_loc","residue_name","chain_id",
            "residue_number","insertion","x_coord","y_coord","z_coord","occupancy",
            "b_factor","element_symbol","charge","model_idx","node_id","residue_id"
        ]
        raw_df = raw_df.reindex(columns=cols)
        pdb_df = pdb_df.reindex(columns=cols)
        rgroup_df = rgroup_df.reindex(columns=cols)

        return raw_df, pdb_df, rgroup_df

    # -----------------------
    # Seleção e centróides
    # -----------------------
    def _select_chains(self) -> List[Chain]:
        assert self.structure is not None, "Estrutura não carregada"
        model = self.structure[self.config.model_index]
        if self.config.chains is None:
            return [ch for ch in model]

        chains_found: List[Chain] = []
        wanted = set(self.config.chains)
        for ch in model:
            if ch.id in wanted:
                chains_found.append(ch)
        if not chains_found and not self.config.allow_empty_chains:
            raise ValueError(f"Nenhuma das cadeias solicitadas foi encontrada: {self.config.chains}")
        return chains_found

    def _collect_residues(self, chains: List[Chain]) -> List[Tuple[str, Residue, np.ndarray]]:
        """Retorna lista de (node_id, residue, centroid)."""
        out: List[Tuple[str, Residue, np.ndarray]] = []
        for ch in chains:
            for res in ch.get_residues():
                if not _is_protein_residue(res):
                    continue
                coords = _heavy_atom_coords(res)
                cent = _centroid(coords)
                out.append((_node_id(ch.id, res), res, cent))
        return out

    def _collect_waters(self, chains: List[Chain]) -> List[Tuple[str, Residue, np.ndarray]]:
        """Retorna lista de (node_id, residue, centroid) para águas HOH."""
        if not self.config.include_waters:
            return []
        out: List[Tuple[str, Residue, np.ndarray]] = []
        for ch in chains:
            for res in ch.get_residues():
                if _is_water(res):
                    coords = _heavy_atom_coords(res)  # heavy átomos da água são O e possivelmente D/None
                    cent = _centroid(coords)
                    out.append((_node_id(ch.id, res, kind="water"), res, cent))
        return out

    # -----------------------
    # Construção do grafo
    # -----------------------
    def build_graph(self) -> BuiltGraph:
        """Pipeline completo: load → selecionar cadeias → ASA/RSA → distâncias → grafo."""
        if self.structure is None:
            self.load()
        chains = self._select_chains()

        # Resíduos e águas
        res_tuples = self._collect_residues(chains)  # [(id, res, centroid)]
        water_tuples = self._collect_waters(chains)  # idem
        raw_pdb_df, pdb_df, rgroup_df = self._make_atom_tables(chains)

        if not res_tuples:
            raise ValueError("Nenhum resíduo proteico encontrado para as cadeias selecionadas")

        # ASA e RSA
        asa_rsa: Dict[str, Tuple[float, float]] = {}
        if self.config.compute_rsa:
            asa_rsa = self._compute_asa_rsa(res_tuples)

        # Montagem de arrays para distâncias
        res_ids = [t[0] for t in res_tuples]
        res_objects = [t[1] for t in res_tuples]
        res_centroids = np.vstack([t[2] for t in res_tuples])

        # Matriz de distâncias resíduo-resíduo
        # Distância Euclidiana: usar broadcasting para obter matriz NxN de forma vetorizada
        diff = res_centroids[:, None, :] - res_centroids[None, :, :]
        dist_mat = np.sqrt(np.sum(diff * diff, axis=2))

        # Grafo
        G = nx.Graph()

        # Adiciona nós de resíduos
        for nid, res, cent in res_tuples:
            asa, rsa = asa_rsa.get(nid, (None, None))
            G.add_node(
                nid,
                kind="residue",
                chain=res.get_parent().id,
                resname=res.get_resname(),
                resseq=int(res.id[1]),
                icode=(res.id[2].strip() or ""),
                centroid=tuple(float(x) for x in cent),
                asa=None if asa is None else float(asa),
                rsa=None if rsa is None else float(rsa),
            )

        # Arestas resíduo-resíduo por cutoff
        cutoff = self.config.residue_distance_cutoff
        N = len(res_ids)
        for i in range(N):
            for j in range(i + 1, N):
                d = float(dist_mat[i, j])
                if d <= cutoff and d > 0.0:
                    G.add_edge(res_ids[i], res_ids[j], distance=d, kind="res-res")

        # Adiciona nós de água e arestas resíduo-água, se habilitado
        water_ids: List[str] = []
        water_centroids: Optional[np.ndarray] = None
        if water_tuples:
            water_ids = [t[0] for t in water_tuples]
            water_centroids = np.vstack([t[2] for t in water_tuples])

            for nid, res, cent in water_tuples:
                hetflag, resseq, icode = res.id
                chain_id = res.get_parent().id  # type: ignore[attr-defined]
                G.add_node(
                    nid,
                    kind="water",
                    chain=chain_id,
                    resname="HOH",
                    resseq=int(resseq),
                    icode=(icode.strip() or ""),
                    centroid=tuple(float(x) for x in cent),
                )

            # Conecta água aos resíduos próximos
            if water_centroids is not None:
                # distâncias água-resíduo: shape (W, R)
                wd = water_centroids[:, None, :] - res_centroids[None, :, :]
                wdist = np.sqrt(np.sum(wd * wd, axis=2))
                wcut = self.config.water_distance_cutoff
                W, R = wdist.shape
                for wi in range(W):
                    for rj in range(R):
                        d = float(wdist[wi, rj])
                        if d <= wcut and d > 0.0:
                            G.add_edge(water_ids[wi], res_ids[rj], distance=d, kind="wat-res")


        built = BuiltGraph(
            graph=G,
            residue_index=list(zip(res_ids, res_objects)),
            residue_centroids=res_centroids,
            water_index=[(t[0], t[1]) for t in water_tuples],
            water_centroids=water_centroids,
            distance_matrix=dist_mat if self.config.store_distance_matrix else None,
        )

        built.raw_pdb_df = raw_pdb_df
        built.pdb_df = pdb_df
        built.rgroup_df = rgroup_df

        return built

    # -----------------------
    # Exportadores
    # -----------------------
    @staticmethod
    def to_graphml(G: nx.Graph, path: str) -> None:
        nx.write_graphml(G, path)
        log.info("GraphML salvo em %s", path)

    @staticmethod
    def to_json(G: nx.Graph, path: str) -> None:
        # Converter numpy types para tipos nativos
        def _convert(v):
            if isinstance(v, (np.floating, np.float32, np.float64)):
                return float(v)
            if isinstance(v, (np.integer, np.int32, np.int64)):
                return int(v)
            if isinstance(v, (np.ndarray,)):
                return v.tolist()
            return v

        data = {
            "nodes": [
                {"id": n, **{k: _convert(v) for k, v in d.items()}}
                for n, d in G.nodes(data=True)
            ],
            "edges": [
                {"u": u, "v": v, **{k: _convert(w) for k, w in d.items()}}
                for u, v, d in G.edges(data=True)
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info("JSON salvo em %s", path)


# Exemplo de uso rápido via CLI python -m pMHC.core.pdb_graph_builder arquivo.pdb A,C
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    ap = argparse.ArgumentParser(description="Gerar grafo de pMHC a partir de PDB/mmCIF")
    ap.add_argument("pdb_path", help="Arquivo PDB ou mmCIF")
    ap.add_argument("--chains", type=str, default=None, help="Cadeias separadas por vírgula, ex: A,C")
    ap.add_argument("--include_waters", action="store_true", help="Incluir moléculas de água como nós")
    ap.add_argument("--res_cut", type=float, default=10.0, help="Cutoff resíduo-resíduo em Å")
    ap.add_argument("--wat_cut", type=float, default=6.0, help="Cutoff água-resíduo em Å")
    ap.add_argument("--no_rsa", action="store_true", help="Não calcular ASA e RSA")
    ap.add_argument("--graphml", type=str, default=None, help="Caminho para salvar GraphML")
    ap.add_argument("--json", type=str, default=None, help="Caminho para salvar JSON")

    args = ap.parse_args()

    cfg = GraphBuildConfig(
        chains=None if args.chains is None else [c.strip() for c in args.chains.split(",") if c.strip()],
        include_waters=args.include_waters,
        residue_distance_cutoff=args.res_cut,
        water_distance_cutoff=args.wat_cut,
        compute_rsa=not args.no_rsa,
    )
    builder = PDBGraphBuilder(args.pdb_path, cfg)
    built = builder.build_graph()

    if args.graphml:
        PDBGraphBuilder.to_graphml(built.graph, args.graphml)
    if args.json:
        PDBGraphBuilder.to_json(built.graph, args.json)

    print(f"Grafo gerado com {built.graph.number_of_nodes()} nós e {built.graph.number_of_edges()} arestas.")
