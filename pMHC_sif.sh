apptainer run pMHC_graph.sif \
  --folder_path $(pwd)/pMHC_graph/pdb_input \
--run_name 4pmhc_identity_teste \
--association_mode identity \
--output_path $(pwd)/pMHC_graph/data/tests/ \
--residues_lists $(pwd)/pMHC_graph/residues_lists.json \
--factors_path  $(pwd)/pMHC_graph/resources/atchley_aa.csv \
--neighbor_similarity_cutoff 0.8 \
--residues_similarity_cutoff 0.8 \
--rsa_similarity_threshold 0.8 \
--rsa_filter 0.1 \
--depth_similarity_threshold 0.8 \
--distance_diff_threshold 3.0 \
--debug True

# --files_name file1,file2,file3...
