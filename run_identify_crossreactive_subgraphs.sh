python3 main.py \
--mols_path pdb_input \
--run_name 4pmhc_identity_teste \
--association_mode similarity \
--output_path data/classI_melyssa/renumbered/withoutTCR/analysis/output5 \
--residues_lists residues_lists.json \
--factors_path  resources/atchley_aa.csv \
--neighbor_similarity_cutoff 0.8 \
--residues_similarity_cutoff 0.6 \
--rsa_similarity_threshold 0.96 \
--debug True
# --interface_list resources/interface_MHC_unique.csv \
#/home/helder/Projects/pMHC_graphs/pdbs_teste/pmhc_mage3_5brz_renumber.pdb \
#/home/helder/Projects/pMHC_graphs/pdbs_teste/pmhc_flu_4nqx_renumber.pdb \
