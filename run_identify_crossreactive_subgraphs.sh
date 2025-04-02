python3 main.py \
--folder_path pdb_input \
--run_name 4pmhc_identity_teste \
--association_mode identity \
--output_path data/tests/ \
--residues_lists residues_lists.json \
--factors_path  resources/atchley_aa.csv \
--neighbor_similarity_cutoff 0.8 \
--residues_similarity_cutoff 0.8 \
--rsa_similarity_threshold 0.8 \
--rsa_filter 0.1 \
--depth_similarity_threshold 0.8 \
--distance_diff_threshold 3.0 \
--debug True
# --interface_list resources/interface_MHC_unique.csv \
#/home/helder/Projects/pMHC_graphs/pdbs_teste/pmhc_mage3_5brz_renumber.pdb \
#/home/helder/Projects/pMHC_graphs/pdbs_teste/pmhc_flu_4nqx_renumber.pdb \
