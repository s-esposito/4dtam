<p align="center">

  <h1 align="center"> 4DTAM: Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians
  </h1>
  <p align="center">
    <a href="https://muskie82.github.io/"><strong>Hidenobu Matsuki</strong></a>
    ·
    <a href="https://www.baegwangbin.com/about"><strong>Gwangbin Bae</strong></a>
    ·
    <a href="https://www.doc.ic.ac.uk/~ajd/"><strong>Andrew J. Davison</strong></a>
  </p>

  <h3 align="center"> CVPR 2025 </h3>



[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center"><a href="https://muskie82.github.io/4dtam/paper/4dtam_paper.pdf">Paper</a> | <a href="https://youtu.be/MRGhggLmTF0?si=IBpJDTrKBHek5DGW">Video</a> | <a href="https://muskie82.github.io/4dtam/">Project Page</a></h3>
  <div align="center"></div>

<p align="center">
  <a href="">
    <img src="./media/teaser.gif" alt="teaser" width="100%">
  </a>
</p>
<p align="center">
This software implements dense SLAM system presented in our paper <a href="https://arxiv.org/abs/2312.06741">4DTAM</a> in CVPR2025, the first full 4D SLAM approach based on Dynamic Surface Gaussian representation optimized through differentiable rendering framework.
</p>
<br>

# Getting Started
## Installation
```
git clone https://github.com/muskie82/4dtam.git --recursive
cd 4dtam
```
Setup the environment.

```
conda env create -f environment.yml
conda activate 4dtam
```

Install the CUDA-dependent libraries:
```
pip install submodules/simple-knn
pip install submodules/diff-surfel-rasterization

pip install --no-build-isolation git+https://github.com/princeton-vl/lietorch.git@0fa9ce8ffca86d985eca9e189a99690d6f3d4df6

pip install "git+https://github.com/facebookresearch/pytorch3d.git"

pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

```

Depending on your setup, please change the dependency version of pytorch/cudatoolkit/lietorch in `environment.yml` by following [this document](https://pytorch.org/get-started/previous-versions/).

Our test setup were:
- Ubuntu 18.04: 
  - `pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3`
  - `lietorch: 0fa9ce8ffca86d985eca9e189a99690d6f3d4df6`

## Try 4DTAM Results
### Dowload 4D reconstruction results
```
bash scripts/download_sample_results.sh
```
Or, directly download the results from [this gdrive link](https://drive.google.com/drive/folders/1Jen2lklrga45e-h1S_sItHr-_R9qXnhp?usp=drive_link)

### Visualize the Results
```
python tools/visualize.py --path sample_results/realsense_data/sloths
```

```
python tools/visualize.py --path sample_results/sim4d/flag_cvpr
```

<p align="center">
  <a href="">
    <img src="./media/gui.gif" alt="teaser" width="70%">
  </a>
</p>
<p align="center">

You can toggle the 4D reconstruction playback using the `Replay` and `Stop` buttons. The visualizer supports various rendering options.

## Run 4DTAM from scratch
### Downloading Datasets
Running the following scripts will automatically download datasets to the `./datasets` folder.
#### Sim4D dataset
In this paper, we introduce a novel synthetic dataset for 4D SLAM research that provides reliable ground-truth data for both camera poses and time-varying scene geometry, two quantities that are difficult to capture with commodity sensors. For full details about the dataset, including the rendering script, please refer to [our separate repository](https://github.com/baegwangbin/sim4d).

```bash
bash scripts/download_sim4d.sh
```
You can manually download Sim4D datasets from [this gdrive link](https://drive.google.com/drive/folders/1XjUF5sTLeHEybmPo6AS_gRNy5eLd2XNU?usp=drive_link)

#### Self-captured Realsense data
```bash
bash scripts/download_realsese_data.sh
```
You can manually download realsense data from [this gdrive link](https://drive.google.com/drive/folders/1owC13xD82atb_etCLRbUtsBkvi03ENt8?usp=drive_link)


### Run online reconstruction
#### Sim4D
```bash
 python 4dtam.py --config configs/sim4d/modular_vehicle.yaml
```

#### Self-captured Realsense data
```bash
 python 4dtam.py --config configs/realsense/linux_book.yaml
 ```

You can disable the GUI and run the method in evaluation mode by adding the `--eval` flag;
 ```bash
 python 4dtam.py --config configs/realsense/linux_book.yaml --eval
```

Finally, you can visualize the processed result by;
```
python tools/visualize.py --path path/to/saved_results
```

# Evaluation
To evaluate our method, first run the method with `--eval` flag  to the command line argument:
```bash
 python 4dtam.py --config configs/sim4d/modular_vehicle.yaml --eval
```
This will save trajectory and 4D map to the directory in config yaml file with `results` item in `Results:save_dir`.

Then, run the evaluation script to get metrics
```bash
 python tools/eval.py --path path/to/saved_results
```

# Acknowledgement
This work incorporates many open-source codes. We extend our gratitude to the authors of the software.
- [Deformable 3D Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussiansg)
- [Dynamic 3D Gaussians](https://github.com/JonathonLuiten/Dynamic3DGaussians)
- [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)
- [Differential Surfel Rasterization
](https://github.com/hbb1/diff-surfel-rasterization)
- [Tiny Gaussian Splatting Viewer](https://github.com/limacv/GaussianSplattingViewer)
- [MonoGS](https://github.com/muskie82/MonoGS)

# License
4DTAM is released under a **LICENSE.md**. For a list of code dependencies which are not property of ours, please check **Dependencies.md**.

# Citation
If you found this code/work to be useful in your own research, please considering citing the following:

```bibtex
@inproceedings{4DTAM,
  title={4DTAM: Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians},
  author={Matsuki, Hidenobu and Bae, Gwangbin and Davison, Andrew},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}

```













