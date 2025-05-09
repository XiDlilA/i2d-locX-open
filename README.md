## 1. Requirements and Installation

The project was developed and evaluated on Ubuntu 20.04, using Python 3.11, PyTorch 2.7.0, and CUDA 11.8.

> Note  
> `visibility_package` has been updated for this release; do not run inference with the original *I2D-Loc* codebase.

### 1.1 Set up the environment

```bash
# create a clean Conda environment
conda create -n i2d-locx python=3.11
conda activate i2d-locx
```

### 1.2 Install PyTorch

Choose the wheel that matches your CUDA version from the official PyTorch “Get Started” page <https://pytorch.org/get-started/locally/>.

Example for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 1.3 Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 1.4 Build the visibility package

```bash
cd pkg/visibility_package
python setup.py install
```

---

## 2. Quick Demo

```bash
bash cmd/sample.sh
```

- **Raw data**: `./i2d-locX-open/sample/0/`  
- **Visualisation results**: `./i2d_locX_sample/test/`