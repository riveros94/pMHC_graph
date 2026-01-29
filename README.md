# Structural Similarity Detection in pMHC Complexes Using Graph-Based Representations

This project implements a graph-based method to identify structurally similar regions across protein complexes, with a particular focus on peptide–MHC (pMHC) structures involved in T cell receptor (TCR) recognition. The method was designed to support the analysis of TCR cross-reactivity by detecting conserved surface regions that may be recognized by the same TCR.

## Overview

TCR cross-reactivity arises from the ability of a single TCR to recognize multiple pMHC complexes that share structurally similar surface regions. Sequence-based approaches are often insufficient to capture these similarities, as they ignore MHC polymorphism and conformational variability.

This project addresses these limitations by representing protein surfaces as graphs and identifying compatible substructures through graph association. Residues are represented as nodes, spatial proximity defines edges, and local surface patterns are encoded using triads of residues. Structural similarity is inferred by assembling compatible triads under geometric and physicochemical constraints.

The method supports:
- Pairwise comparison of pMHC structures
- Identification of shared surface regions
- Explicit separation of MHC- and peptide-associated nodes
- Analysis across different MHC alleles and peptide lengths

## Method Summary

1. **Graph Construction**  
   Protein structures are converted into residue-level graphs using the Graphein library. Nodes represent residues filtered by solvent accessibility and depth, and edges are defined based on Euclidean distance thresholds.

2. **Triad Encoding**  
   Local surface patterns are represented as residue triads. Each triad can encode:
   - Inter-residue distances (discretized)
   - Relative solvent accessibility (RSA)
   - Optional residue class groupings

3. **Graph Association and Frame Generation**  
   Compatible triads across structures are associated under geometric consistency constraints, including limits on distance variation. These associations are assembled into frames representing structurally similar surface regions.

4. **Structural Mapping**  
   Identified regions can be visualized as graphs or mapped back onto 3D structures, highlighting residues involved in potential cross-reactive interfaces.

## Requirements

- **Python 3.10.8** 
- **networkx**
- **numpy**
- **pandas**
- **biopython**
- **matplotlib**
- **scikit-learn**
- **plotly**
- **pyvis**
- **memory_profiler**
- **gemmi**

## Running the Pipeline

The pipeline is configured via a JSON manifest file, which defines execution settings, input structures, and residue selectors. The manifest acts as the primary interface for configuring all aspects of the analysis, including graph construction, triad encoding, and similarity constraints.

## Key Parameters

| Parameter | Description | Type / Values |
|----------|------------|---------------|
| run_name | Identifier for the execution run | string |
| output_path | Output directory | string |
| track_steps | Save intermediate steps | boolean |
| rsa_table | Reference ASA table | Sander, Wilke, Miller |
| edge_threshold | Distance cutoff for graph edges (Å) | float |
| node_granularity | Atomic representation | all_atoms, ca_only, backbone, sidechain |
| triad_rsa | Include RSA in triad tokens | boolean |
| rsa_filter | Minimum RSA for residue inclusion | float |
| rsa_bin_width | Width of RSA discretizated bins | float |
| distance_bin_width | Width of distance bins (Å) | float |
| distance_std_threshold | Max allowed distance variation within frames | float |
| distance_diff_threshold | Max allowed distance difference between triads | float |
| include_noncanonical | Include ligands or waters as nodes | boolean |
  
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
