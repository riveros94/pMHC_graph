import logging
import sys
from itertools import combinations
from pathlib import Path

from immunograph.cli.cli_parser import parse_args
from immunograph.workflow.manifest import load_manifest, build_association_config
from immunograph.workflow.association import run_association_task 
from immunograph.utils.preprocessing import create_graphs
from immunograph.core.tracking import init_tracker

def main():
    args = parse_args() 
    manifest = load_manifest(args.manifest)
    S = manifest["settings"] 
    
    base_run_name = S["run_name"]
    base_output_path = Path(S["output_path"])
    run_mode = S.get("run_mode", "all") 

    init_tracker(
        root="CrossSteps",
        outdir=base_run_name,
        enabled=S.get("track_steps", False),
        prefer_npy_for_ndarray=True,
        add_timestamp_prefix=False,
    )

    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if S.get("debug", False) else logging.INFO
    )
    log = logging.getLogger("CRSProtein")
    log.setLevel(logging.DEBUG if S.get("debug", False) else logging.INFO)

    graphs = create_graphs(manifest)
    base_association_config = build_association_config(S, run_mode)

    if run_mode == "all":
        target_dir = base_output_path / "ALL"
        run_association_task(
            graphs=graphs,
            output_path=target_dir,
            run_name=base_run_name, 
            association_config=base_association_config,
            log=log
        )

    elif run_mode == "pair":
        pair_base_dir = base_output_path / "PAIR"
        
        for g1, g2 in combinations(graphs, 2):
            name1 = Path(g1[1]).stem
            name2 = Path(g2[1]).stem
            
            name1 = name1.replace("_nOH", "")
            name2 = name2.replace("_nOH", "")

            pair_folder_name = f"{name1}_vs_{name2}"
            target_dir = pair_base_dir / pair_folder_name
            
            pair_run_name = f"{base_run_name}_{name1}_{name2}"

            run_association_task(
                graphs=[g1, g2],
                output_path=target_dir,
                run_name=pair_run_name,
                association_config=base_association_config,
                log=log
            )
            
    else:
        log.error(f"Unknown run_mode: {run_mode}. Please use 'all' or 'pair'.")

if __name__ == "__main__":
    main()
