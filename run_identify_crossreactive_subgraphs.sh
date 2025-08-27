python3 main.py \
--folder_path /home/elementare/GithubProjects/pMHC_graph/Analysis/selected_strs_renumber/without_TCR/ \
--files_name noTCR_3tjh.trunc.fit_renum.pdb,noTCR_3tfk.trunc.fit_renum.pdb \
--run_name 3tjh_3tfk_ca_only \
--association_mode identity \
--output_path data/3tjh_3tfk_ca_only/ \
--manifest jsons/residues_lists.json \
--factors_path  resources/atchley_aa.csv \
--rsa_similarity_threshold 0.7 \
--rsa_filter 0.1 \
--depth_filter 10 \
--depth_similarity_threshold 0.7 \
--distance_diff_threshold 10.0 \
--centroid_threshold 11 \
--check_rsa True \
--check_depth False \
--distance_bins 3 \
--rsa_bins 3 \
--debug True \
--track_steps True \
--centroid_granularity "ca_only" \
--exclude_waters False \
# --classes_path jsons/residues_classes.json \
# --interface_list resources/interface_MHC_unique.csv \
#/home/helder/Projects/pMHC_graphs/pdbs_teste/pmhc_mage3_5brz_renumber.pdb \
#/home/helder/Projects/pMHC_graphs/pdbs_teste/pmhc_flu_4nqx_renumber.pdb \
