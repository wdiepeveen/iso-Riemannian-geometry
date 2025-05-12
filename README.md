# iso-Riemannian geometry

    [1] W. Diepeveen, D. Needell.  
    Manifold Learning with Normalizing Flows: Towards Regularity, Expressivity and Iso-Riemannian Geometry
    arXiv preprint arXiv:xxxx.xxxx. 2025 MMM DD.

Setup
-----

The recommended (and tested) setup is based on Python 3.8. Install the following dependencies with anaconda:

    # Create conda environment
    conda create --name irg python=3.8
    conda activate irg

    # Clone source code and install
    git clone https://github.com/wdiepeveen/iso-Riemannian-geometry.git
    cd "iso-Riemannian-geometry"
    pip install -r requirements.txt


Reproducing the experiments in [1]
----------------------------------

To produce the results in [1]. 
* For the double gaussian data results run:
  *  `double_gaussian_affine_unbend.ipynb`,
  *  `double_gaussian_affine_anisotropic_nflow.ipynb`,
  *  `double_gaussian_additive_nflow.ipynb`.
* For the hemisphere data results run `sphere_additive_nflow.ipynb`.
* For the mnist data results run `mnist_additive_nflow.ipynb`.
