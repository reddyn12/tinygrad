env_name=MsPacman

python3 -u train.py \
    -n "${env_name}-life_done-wm_2L512D8H-100k-seed1" \
    -seed 1 \
    -config_path "config_files/STORM.yaml" \
    -env_name "ALE/${env_name}-v5" \
    -trajectory_path "D_TRAJ/${env_name}.pkl" 
# CUDA=1 HALF=1 python3 -u train.py \
#     -n "${env_name}-life_done-wm_2L512D8H-100k-seed1" \
#     -seed 1 \
#     -config_path "config_files/STORM.yaml" \
#     -env_name "ALE/${env_name}-v5" \
#     -trajectory_path "D_TRAJ/${env_name}.pkl" 

# python3 -u train.py -n "${env_name}-life_done-wm_2L512D8H-100k-seed1" -seed 1 -config_path "config_files/STORM.yaml" -env_name "ALE/${env_name}-v5" -trajectory_path "D_TRAJ/${env_name}.pkl" 