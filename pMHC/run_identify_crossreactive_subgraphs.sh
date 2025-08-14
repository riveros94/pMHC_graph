python3 main.py \
--folder_path pdb_input \
--run_name Mage3xTitin \
--association_mode identity \
--output_path data/Mage3xTitin_8_5/ \
--residues_lists jsons/residues_lists.json \
--factors_path  resources/atchley_aa.csv \
--rsa_similarity_threshold 0.7 \
--rsa_filter 0.1 \
--depth_filter 10 \
--depth_similarity_threshold 0.7 \
--distance_diff_threshold 10.0 \
--centroid_threshold 8.5 \
--check_rsa True \
--check_depth False \
--distance_bins 3 \
--rsa_bins 3 \
--debug True
# --classes_path jsons/residues_classes.json \
# --interface_list resources/interface_MHC_unique.csv \
#/home/helder/Projects/pMHC_graphs/pdbs_teste/pmhc_mage3_5brz_renumber.pdb \
#/home/helder/Projects/pMHC_graphs/pdbs_teste/pmhc_flu_4nqx_renumber.pdb \
