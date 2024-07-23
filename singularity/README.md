# Singularity

```sh
# Create a 16GB swap file
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
# Verify that the swap file is active
sudo swapon --show
# Create temporary directories for singularity
mkdir cache tmp tmp2
```

```sh
SINGULARITY_TMPDIR=`pwd`/tmp SINGULARITY_CACHEDIR=`pwd`/cache sudo singularity build --tmpdir `pwd`/tmp2 torch2.3.1-torch_geometric-cuda12.1-cudnn8.sif Singularity.def
```

```sh
sudo singularity exec --nv ./torch2.3.1-torch_geometric-cuda12.1-cudnn8.sif /bin/bash
```