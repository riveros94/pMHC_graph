
# Common Subgraph Detection in Protein Structures

This project identifies common subgraphs in protein structures using the Graphein library. It can be used with any protein structures, mapping common subgraphs to highlight areas of structural similarity between them.

## Overview

The algorithm is inspired by work on protein-protein binding sites and surface structure conservation. It analyzes the input protein structures to detect common subgraphs and map them onto the structural representation of the proteins.

### Requirements
- **Python:** Tested with Python 3.8
- **Graphein:** Make sure Graphein is installed in a Conda environment.

### Known Issues
- Graphein's centroid computation issue with multiple chains was resolved by modifying the file at:
  `graphein/protein/graphs.py`
  
  Replace:
  ```python
  ["residue_number", "chain_id", "residue_name", "insertion"]
  ```
  with:
  ```python
  ["chain_id", "residue_number", "residue_name", "insertion"]
  ```

- Compatibility issues when running DSSP with Graphein were resolved by modifying the file at:
  `graphein/protein/features/nodes/dssp.py`

  Change:
  ```python
  pdb_file, DSSP=executable, dssp_version=dssp_version
  ```
  to:
  ```python
  pdb_file, DSSP=executable
  ```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/LBC-LNBio/pMHC_graph.git
    ```

2. Create a Conda environment and install the required packages:
    ```bash
    conda create -n graphein_env python=3.8
    conda activate graphein_env
    pip install -r requirements.txt
    ```

3. Install Graphein:
    ```bash
    pip install graphein
    ```

## Usage

### Command Example

Run the script with the following command:
```bash
python3 main.py --mols_path /path/to/pdb_folder --residues_lists /path/to/residues_list.json --centroid_threshold 10 --run_name test_run --association_mode similarity --output_path ./output_directory
```

### Arguments

| Argument                     | Description                                                 | Default             |
|------------------------------|-------------------------------------------------------------|---------------------|
| `--mols_path`                | Path with PDB input files.                                  | `''`                |
| `--centroid_threshold`       | Distance threshold for building the interface graphs.       | `10`                |
| `--run_name`                 | Name for storing results in the output folder.              | `test`              |
| `--association_mode`         | Mode for creating association nodes (identity/similarity).  | `identity`          |
| `--output_path`              | Path to store output results.                              | `'~/'`              |
| `--neighbor_similarity_cutoff` | Threshold for neighbor's similarity.                     | `0.95`              |
| `--rsa_filter`               | Threshold for filtering residues by RSA.                   | `0.1`               |
| `--rsa_similarity_threshold` | Threshold for RSA similarity in association graphs.        | `0.95`              |
| `--residues_lists`           | Path to JSON file containing the pdb residues.             | `None`              |
| `--debug`                    | Activate debug mode.                                       | `False`             |

## Output

- **Common Subgraphs:** The script generates common subgraphs that are mapped to the protein structures.
- **Visualization:** Graphs of the common subgraphs are saved as images in the output directory.

## Logging

The script supports debug mode for detailed logging. Activate debug mode by setting the `--debug` flag to `True`.

## References

- **Original Algorithm:** The algorithm is inspired by methods used in protein-protein binding sites and surface structure analysis.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Graphein Library:** Used for protein graph analysis.
