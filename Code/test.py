import torch
import torch_geometric
def PyTorch_CUDA_Check():
    """
        Check PyTorch and CUDA Whether Available
    """
    print('PyTorch Version:', torch.__version__)
    print('CUDA Whether Available:', torch.cuda.is_available())
    print('CUDA Version:', torch.version.cuda)
    print('GPU Name:', torch.cuda.get_device_name(0))



def PyG_Check():
    """
    Check PyTorch Geometric and its core libraries
    """
    print('=' * 60)
    print('PyG Env Check')
    print('=' * 60)
    
    # PyTorch info
    print('PyTorch Version:', torch.__version__)
    print('CUDA Available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA Version:', torch.version.cuda)
        print('GPU Name:', torch.cuda.get_device_name(0))
    print()
    
    # PyG main library
    print('PyG Version:', torch_geometric.__version__)
    
    # Four underlying libraries
    try:
        import torch_scatter
        print('torch-scatter:', torch_scatter.__version__)
    except:
        print('torch-scatter: ❌ UnInstalled')
    
    try:
        import torch_sparse
        print('torch-sparse:', torch_sparse.__version__)
    except:
        print('torch-sparse: ❌ UnInstalled')
    
    try:
        import torch_cluster
        print('torch-cluster:', torch_cluster.__version__)
    except:
        print('torch-cluster: ❌ UnInstalled')
    
    try:
        import torch_spline_conv
        print('torch-spline-conv:', torch_spline_conv.__version__)
    except:
        print('torch-spline-conv: ❌ UnInstalled')
    
    print('=' * 60)
    print('✅ PyG Env Check! All libraries are installed!')

if __name__ == "__main__":
    PyTorch_CUDA_Check()#check PyTorch and CUDA Whether Available
    PyG_Check()#check PyTorch Geometric and its core libraries