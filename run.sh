#!/bin/bash
#SBATCH --job-name=lime-sampling
#SBATCH --account=mi2lab-normal
#SBATCH --partition=short
#SBATCH --time=1-00:00:00
#SBATCH --constraint=dgx 
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6GB
#SBATCH --output=logs/%j.log 

set -e
hostname; pwd; date

source .env
module load anaconda/4.0
source $CONDA_SOURCE
conda init
# eval "$(conda shell.bash hook)"
conda activate lime-sampling

if [ $# -lt 1 ]; then
  echo "Usage: $0 <script> [arg1] [arg2]..."
  exit 1
fi

mkdir -p logs/

script_name=$1
extension="${script_name##*.}"
# shift arguments to leave only the command line arguments for the script
shift

if [ "$extension" == "py" ]; then
    python $script_name "$@"
elif [ "$extension" == "sh" ]; then
    bash $script_name "$@"
else
    echo "Unsupported file format: .$extension"
    exit 1
fi