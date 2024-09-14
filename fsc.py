import torch, numpy as np

def make_rfreq_radius_new(tensor1_shape: torch.Tensor):
    """
    Make our frequency grid using rfftfreq for the last axis
    """
    print('tensor1_shape in make radius', tensor1_shape)
    meshgrid = torch.meshgrid([torch.fft.fftfreq(tensor1_shape[0]),
                                torch.fft.rfftfreq(tensor1_shape[1])],
                                indexing = 'ij')
    print('meshgridshape', meshgrid[0].shape, meshgrid[1].shape)
    freq_radius = torch.sqrt(torch.pow(meshgrid[0],2) + torch.pow(meshgrid[1],2))
    print(torch.max(freq_radius), 'max radius', freq_radius.shape)
    del(meshgrid)
    return freq_radius

def _normalised_cc_real_1d(tensor1: torch.Tensor, tensor2:torch.Tensor):
    correlation = torch.dot(tensor1, torch.conj(tensor2))
    return correlation / (torch.linalg.norm(tensor1) * torch.linalg.norm(tensor2))

def _get_spacing(tensor_shape: torch.Tensor):
    if len(tensor_shape) == 2:
        return np.min(np.array([np.cos(np.arctan(tensor_shape[1]/ tensor_shape[0])),
                                np.sin(np.arctan(tensor_shape[1]/ tensor_shape[0]))]))
    elif len(tensor_shape == 3):
        return np.min(np.array([np.cos(np.arctan(np.cos(np.arctan(tensor_shape[2]/ tensor_shape[0])),
                                np.sin(np.arctan(tensor_shape[1]/ tensor_shape[0]))]))
    np.cos(np.arctan(1))

def compute_rfsc(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """Computes the FSC between tensor1 and tensor2 which need not be square"""
    rfft_spacing = _get_spacing(tensor1.shape)
    freq_radius = make_rfreq_radius_new(torch.clone(torch.as_tensor(tensor1.shape)))
    max_dist = torch.sqrt(torch.tensor((tensor1.shape[0]/2)**2 + (tensor1.shape[1]/2)**2))

    tensor1 = torch.fft.rfft2(tensor1)
    tensor2 = torch.fft.rfft2(tensor2)
    assert tensor1.shape == tensor2.shape
    sorted_frequencies_flat, sort_idx_flat = torch.sort(torch.flatten(freq_radius), descending = False)
    del(freq_radius)
    #spacing allows our spherical shell to encompass all data instead of the largest centered sphere, 
    #ie a spherical slice out of a thin rectanular prism
    bins = torch.fft.rfftfreq(torch.ceil(max_dist).int(), d = rfft_spacing)
    #spacing allows up to get higher frequencies
    bin_interval = 1 / torch.ceil(max_dist) #bin_centers[1] - bin_centers[0]
    #skipping is our binning distance
    binning_factor = 2
    bin_centers = torch.cat([bins, torch.as_tensor([bin_centers[-1] + bin_interval])])[::binning_factor] 
    bin_centers = bin_centers.unfold(dimension=0, size=2, step=1)
    bin_centers = torch.mean(bin_centers, dim = 1)
    split_idx   = torch.searchsorted(sorted_frequencies_flat, bin_centers)
    shell_index = torch.tensor_split(sort_idx_flat, split_idx)[:-1] #exlude last one because of binning
    print(len(shell_index), 'shell shape')
    #del(sort_idx_flat)
    tensor1 = torch.flatten(tensor1)
    tensor2 = torch.flatten(tensor2)
    for idx in shell_index[-10:]:
        print(tensor1[idx].shape, tensor2[idx].shape, 'last10 shapes')

 
    fsc = torch.tensor([
    _normalised_cc_real_1d(tensor1[idx], tensor2[idx])
    for idx in shell_index
    ])

    return fsc