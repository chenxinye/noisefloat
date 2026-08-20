#!/bin/bash
#SBATCH --job-name=exp7
#SBATCH --partition=convergence
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100_7g.80gb:1
#SBATCH --time=72:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err


set -euo pipefail
cd "$SLURM_SUBMIT_DIR"


module load cuda/13.0
module load cudnn/9.20.0_cuda13

which python
python --version
nvidia-smi


rm -rf output_numpy.txt output_torch.txt
python cadna_examples_1_7.py >> output_numpy.txt
echo "NumPy Job finished at $(date)"
python cadna_examples_1_7_torch.py --device cuda >> output_torch.txt
echo "Torch Job finished at $(date)"

