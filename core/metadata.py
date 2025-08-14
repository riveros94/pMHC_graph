# core/metadata.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, Iterable
from statistics import mean

def _graph_mean(values):
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else None

def rsa(G, **ctx):
    """
    Função de metadados de GRAFO que resume RSA já presente nos nós.
    Este pipeline supõe que o cálculo de RSA foi feito no builder.
    Aqui computamos estatísticas de grafo e salvamos em G.graph.
    """
    rsa_vals = [d.get("rsa") for _, d in G.nodes(data=True) if d.get("kind") != "water"]
    G.graph["rsa_mean"] = _graph_mean(rsa_vals)
    G.graph["rsa_count_nonnull"] = sum(1 for x in rsa_vals if x is not None)
    # proporção de nós bem expostos, por exemplo rsa >= 0.25
    G.graph["rsa_prop_exposed_025"] = (
        sum(1 for x in rsa_vals if x is not None and x >= 0.25) / len(rsa_vals)
    ) if rsa_vals else None
    return G

def secondary_structure(G, **ctx):
    """
    Anota estrutura secundária via DSSP se for possível.
    Espera encontrar em ctx:
      - 'residue_map': dict {node_id: Bio.PDB.Residue}
      - 'dssp_config': DSSPConfig
      - 'structure': Bio.PDB.Structure

    Se DSSP não estiver habilitado ou faltar contexto, simplesmente retorna G.
    """
    dssp_cfg = ctx.get("dssp_config")
    structure = ctx.get("structure")
    residue_map = ctx.get("residue_map")  # {node_id: Residue}

    if not dssp_cfg or not dssp_cfg.enabled or structure is None or not residue_map:
        return G

    try:
        from Bio.PDB.DSSP import DSSP
    except Exception:
        # Biopython sem DSSP instalado
        return G

    # DSSP por modelo 0
    model = structure[0]
    try:
        if dssp_cfg.dssp_path:
            dssp = DSSP(model, ctx.get("pdb_path"), dssp=dssp_cfg.dssp_path)
        else:
            dssp = DSSP(model, ctx.get("pdb_path"))
    except Exception:
        # Falha em rodar DSSP, não interrompe pipeline
        return G

    # Mapear para os nós
    for nid, data in G.nodes(data=True):
        if data.get("kind") == "water":
            continue
        res = residue_map.get(nid)
        if res is None:
            continue
        chain_id = res.get_parent().id
        hetflag, resseq, icode = res.id
        key = (chain_id, res.id)

        try:
            dssp_key = (chain_id, res.id)
            ss = dssp[dssp_key][2]  # coluna de SS
        except Exception:
            try:
                dssp_key = (chain_id, (' ', resseq, icode))
                ss = dssp[dssp_key][2]
            except Exception:
                ss = None
        if ss is not None and ss == ' ':
            ss = 'C'  # coil
        G.nodes[nid]["ss"] = ss

    ss_vals = [d.get("ss") for _, d in G.nodes(data=True) if d.get("kind") != "water"]
    counts = {}
    for s in ss_vals:
        counts[s] = counts.get(s, 0) + 1
    G.graph["ss_counts"] = counts
    return G
