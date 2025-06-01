# Gaussian Splatting Regularization for Improved Out-of-Distribution View Synthesis

This repository provides the code for the Master's Thesis "Gaussian Splatting Regularization for Improved
Out-of-Distribution View Synthesis". A link to the paper will be provided soon!

## Demo

https://github.com/user-attachments/assets/ec36259e-2244-4f81-84ca-2f21a220231a


## Quick Start

First, install all the dependencies and install the python package.

```
# First, create conda enviroment
conda create -n oodgs python=3.11 -y
conda activate oodgs

# Assumes CUDA 12.6
./setup.sh
```

To generate the error map using Met3r loss, use:
```
python render_met3r.py
```

To generate the error map using Depth MVC loss, use:
```
python render_depth_mvc.py
```

## Benchmarking

For benchmarking the inference speed (avg forward pass time), use:

```
python benchmark_inference_speed.py {met3r | depth_mvc} --runs <N> --image-size <H>
```

For benchmarking the VRAM usage, use:

```
python benchmark_vram_usage.py {met3r | depth_mvc} --image-size <H>
```

## About 

This project tries to improve the quality of Out-Of-Distribution (OOD) rendering in Gaussian Splatting models. The approach we chose is to regularize the Gaussian optimization by adding OOD regularization in the training pipeline. The illustration of our approach is shown in the image:
![Overall pipeline](media/ood_pipeline.pdf)
