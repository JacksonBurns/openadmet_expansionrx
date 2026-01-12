conda create --name minimol 'python==3.10.*' --yes
conda activate minimol
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129 'numpy<2'
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
pip install minimol 'torch==2.8.0' 'numpy<2' 'scipy==1.12.*' # force minimol to respect the installed version of pytorch and numpy
