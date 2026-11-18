import random
import torch
import numpy
import sys
import datetime
import shutil # for file management
import os
import re
import yaml

# Absolute path to the repo root (directory containing main.py)
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add the repository root to the Python path
sys.path.insert(0, REPO_ROOT)

import algorithms.NSGAII as NSGAII

if __name__ == '__main__':
    assert len(sys.argv) == 9, "Correct usage: python alg_name domain_name data_dirpath alg_config env_config seed label traj_write_freq"
   
    # Process the command line args
    alg_name = sys.argv[1]
    assert alg_name in ['nsga2'], "Unrecognised alg_name"
    domain_name = sys.argv[2]
    assert domain_name in ['rover'], 'Uncrecognised domain_name'
    data_dir = sys.argv[3]
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(REPO_ROOT, data_dir)
    data_dir = data_dir+'/' if data_dir[-1]!='/' else data_dir # Add a directory '/' at the end
    src_alg_config_filename = sys.argv[4]
    if not os.path.isabs(src_alg_config_filename):
        src_alg_config_filename = os.path.join(REPO_ROOT, src_alg_config_filename)
    src_env_config_filename = sys.argv[5]
    if not os.path.isabs(src_env_config_filename):
        src_env_config_filename = os.path.join(REPO_ROOT, src_env_config_filename)
    seed_val = int(sys.argv[6])
    seed_val_str = str(seed_val)
    traj_write_freq = int(sys.argv[7])

    # Datetime for file naming
    datetime_now_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(src_env_config_filename, 'r') as f:
        env_config = yaml.safe_load(f)

    num_agents = env_config['Agents']['num_agents']

    poi_mode = env_config['POI_Generation']['mode']

    if poi_mode == 'fixed':
        num_pois = len(env_config['Environment']['pois'])
    elif poi_mode == 'concentric_rings':
        rings = env_config['POI_Generation']['rings']
        num_pois = sum(ring['num_pois'] for ring in rings)
    else:
        raise ValueError(f"Unrecognized POI generation mode: {poi_mode}")

    # Sanitize label for filesystem safety
    safe_label = f"A{num_agents}_P{num_pois}_S{seed_val_str}"

    
    # Find existing runs with the same label
    existing = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and re.match(rf"^{re.escape(safe_label)}_\d+$", d)
    ]

    # Extract run numbers
    run_nums = [
        int(d.split('_')[-1]) for d in existing
    ]

    next_run_num = max(run_nums) + 1 if run_nums else 1
    
    # Create a run-specific subdirectory
    run_id = f"{safe_label}_{next_run_num}"
    run_dir = os.path.join(data_dir, run_id)
    os.makedirs(run_dir, exist_ok=False)
    print(f"Data will be saved to directory: {run_dir}")
    
    # Create copy of configs at save data location
    dest_alg_config_filename = os.path.join(run_dir, 'algconfig.yaml')
    shutil.copyfile(src_alg_config_filename, dest_alg_config_filename)
    dest_env_config_filename = os.path.join(run_dir, 'envconfig.yaml')
    shutil.copyfile(src_env_config_filename, dest_env_config_filename)
    
    # Save the metadata
    metadata_path = os.path.join(run_dir, 'metadata.txt')

    with open(metadata_path, 'w') as f:
        f.write(f"algorithm: {alg_name}\n")
        f.write(f"domain: {domain_name}\n")
        f.write(f"seed: {seed_val}\n")
        f.write(f"label: {safe_label}\n")
        f.write(f"run_number: {next_run_num}\n")
        f.write(f"traj_write_freq: {traj_write_freq}\n")
        f.write(f"datetime: {datetime_now_string}\n")
        f.write(f"alg_config_source: {src_alg_config_filename}\n")
        f.write(f"env_config_source: {src_env_config_filename}\n")

    # Save data filename
    data_filename = os.path.join(run_dir, 'savedata.csv')
    
    # Set the seed value for all libraries
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    numpy.random.seed(seed_val)

    # Initialise the algorithm based on alg name
    alg = NSGAII.NSGAII(alg_config_filename=dest_alg_config_filename,
                        domain_name=domain_name,
                        data_filename=data_filename,
                        rover_config_filename=dest_env_config_filename)

    # Run the algorithm
    next_print = 0.0  # percent
    num_gens = alg.num_gens

    for gen in range(num_gens):
        percent = (gen / num_gens) * 100

        if percent >= next_print:
            print(f"Progress: {int(next_print)}% ({gen}/{num_gens})")
            next_print += 1.0
        alg.evolve(gen=gen, traj_write_freq=traj_write_freq)
