# Gaussian Splatting Regularization for Improved Out-of-Distribution View Synthesis

This repository provides the code for the Master's Thesis "Gaussian Splatting Regularization for Improved
Out-of-Distribution View Synthesis". 

## Demo

<video src="media/ood-gs-kitti-demo.mp4" controls width="100%"></video>



## Quick Start

First, install all the dependencies and install the python package.

```
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


