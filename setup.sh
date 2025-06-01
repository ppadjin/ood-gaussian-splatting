
# Assumes CUDA 12.6

# Install dependencies
pip install -e .

# Install met3r
git submodule update --init --recursive
cd met3r && pip install -e . && cd ..