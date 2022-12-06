   Install first pytorch and its extensions on GPU:
   - Limit to cu111 requirements of pytorch_sparse
   ```sh
    conda install -c conda-forge cudatoolkit=11.1 pytorch-gpu=1.8.0 
    conda install gpytorch
    # conda install pytorch_sparse -c conda-forge
    # conda install pytorch-sparse -c pyg
    # pip install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
    # pip install sparselinear
   
   ```
   If you don't have GPU available install it on CPU:
   ```sh
   conda install -c conda-forge pytorch-cpu
   # pip install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
   # pip install sparselinear
   ```
       
2. 
   ``sh
   conda env create -f environment_without_pytorch.yml
   conda env create -f environment_with_pytorch.yml
   ```
   
   # To customize your enviroment, you can export it using  
   ``sh
   conda env export > environment_without_pytorch.yml    
   conda env export > environment_with_pytorch.yml   