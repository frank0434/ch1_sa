#!/bin/bash
# --------- Name of the job --------------
#SBATCH --job-name=McCain_GSA_Sobol_ProcessingDVS
# --------- Mail address -----------------
###SBATCH --mail-user=jian1.liu@wur.nl
###SBATCH --mail-type=ALL
# --------- Output files ------------------
#SBATCH --output=/home/WUR/liu283/GitRepos/ch1_sa/logs/%x.%J.out    
#SBATCH --error=/home/WUR/liu283/GitRepos/ch1_sa/logs/%x.%J.err   
# --------- Other information -------------
#SBATCH --comment=
# --------- Required resources ------------
#SBATCH --time=0-24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
# ---------- --nodelist=node274-----------
# --------- Environment , Operations and Job step -------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py3_pcse
GSA_sample_sizes=(32768)
# 32 64 125 256 512 1024 2048 4096 8192 16384 32768 65536

for GSA_sample_size in ${GSA_sample_sizes[@]}; do
    python /home/WUR/liu283/GitRepos/ch1_sa/scripts/run_process_sim_res.py --GSA_sample_size $GSA_sample_size  --CPUs $SLURM_CPUS_PER_TASK
    # python /home/WUR/liu283/GitRepos/ch1_sa/scripts/run_calculate_Si.py --GSA_sample_size $GSA_sample_size  --CPUs $SLURM_CPUS_PER_TASK
done
# python /home/WUR/liu283/GitRepos/ch1_LSA/scripts/run_process_sim_res.py
