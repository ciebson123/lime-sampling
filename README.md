# lime-sampling

Docs link: https://docs.google.com/document/d/1WxhNxzkNHw9EIz5GGiMFWAmiM4R3e9kHiILhwoRlBxQ/edit?tab=t.0#heading=h.tem6cjs4aq4f


Datasets - all dataset code (e.g. unified interface and concrete datasets) should be in this folder
Models - all models code and concrete implementations should go here

## Setup

Clone the repository to the local home directory.
```
git clone https://github.com/ciebson123/lime-sampling.git
cd lime-sampling
```

Setup anaconda.
```
module load anaconda/4.0
source $CONDA_SOURCE
echo "$(which conda)"
conda init
conda activate
```

Install dependencies.
```
conda create -n lime-sampling python=3.12.4
conda activate lime-sampling
pip install -r requirements.txt
```

Create directory within shared group space for storing large files and save it in a .env file.
```
PROJECT_DIR=/mnt/evafs/groups/mi2lab/<name>/lime-sampling
mkdir -p $PROJECT_DIR
echo "export PROJECT_DIR=${PROJECT_DIR}" > .env
```


## Running the code

```
# may require changing arguments in the script
sbatch run.sh scripts/calculate_local_explanations.py
```
