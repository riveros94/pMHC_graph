# core/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Iterable, List, Optional, Union
from pathlib import Path

# Apenas para tipagem, import real ocorre no pipeline
try:
    import networkx as nx  # type: ignore
except Exception:
    nx = Any  # fallback de tipo

Granularity = Union[str]  # "centroids", "CA", "atom"

@dataclass
class DSSPConfig:
    """
    Configuração simples para DSSP.
    Se mkdssp estiver no PATH, normalmente não é preciso configurar nada.
    Caso contrário, aponte para o executável com dssp_path.
    """
    dssp_path: Optional[str] = None
    enabled: bool = True  # permite desligar DSSP sem mexer nos metadados

@dataclass
class ProteinGraphConfig:
    """
    Config de construção de grafo proteico no estilo 'Graphein-like', minimalista.

    - granularity: controla a posição dos nós. Neste pipeline usamos "centroids"
      para resíduos. Futuramente dá para suportar "CA" ou "atom".
    - edge_construction_functions: lista de callables (G, **ctx) -> G
    - node_metadata_functions: lista de callables (G, **ctx) -> G
    - edge_metadata_functions: lista de callables (G, **ctx) -> G
    - graph_metadata_functions: lista de callables (G, **ctx) -> G
    - dssp_config: se habilitado, as funções de metadados podem usar DSSP
    """
    granularity: Granularity = "centroids"
    # Conjunto de cadeias alvo. Se None, pipeline usa todas as cadeias.
    chains: Optional[Iterable[str]] = None
    include_waters: bool = True
    residue_distance_cutoff: float = 10.0
    water_distance_cutoff: float = 6.0
    compute_rsa: bool = True

    # Funções de construção e anotação
    protein_df_processing_functions: Optional[List[Callable]] = None
    edge_construction_functions: List[Callable] = field(default_factory=list)
    node_metadata_functions: Optional[List[Callable]] = None
    edge_metadata_functions: Optional[List[Callable]] = None
    graph_metadata_functions: Optional[List[Callable]] = None

    dssp_config: Optional[DSSPConfig] = field(default_factory=DSSPConfig)

def make_default_config(centroid_threshold: float) -> ProteinGraphConfig:
    """
    Devolve um ProteinGraphConfig pronto para uso, compatível com o trecho que você enviou.
    """
    from functools import partial
    from core.edges import add_distance_threshold
    from core.metadata import rsa, secondary_structure

    return ProteinGraphConfig(
        granularity="centroids",
        edge_construction_functions=[
            partial(
                add_distance_threshold,
                threshold=centroid_threshold,
                long_interaction_threshold=0
            )
        ],
        graph_metadata_functions=[rsa, secondary_structure],
        dssp_config=DSSPConfig(),
        include_waters=True,
        compute_rsa=True,
        
    )
