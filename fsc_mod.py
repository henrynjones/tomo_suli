import torch, numpy as np

def make_rfreq_radius_new(tensor1_shape: torch.Tensor):
    """
    Make our frequency grid using rfftfreq for the last axis
    """
    if len(tensor1_shape) == 2:
        freq_radius = torch.meshgrid([torch.fft.fftfreq(tensor1_shape[0]),
                                    torch.fft.rfftfreq(tensor1_shape[1])],
                                    indexing = 'ij')
        freq_radius = torch.sqrt(torch.pow(freq_radius[0],2) + torch.pow(freq_radius[1],2))
    if len(tensor1_shape) == 3:
        freq_radius = torch.meshgrid([torch.fft.fftfreq(tensor1_shape[0]),
                                      torch.fft.fftfreq(tensor1_shape[1]),
                                    torch.fft.rfftfreq(tensor1_shape[2])],
                                    indexing = 'ij')
        freq_radius = torch.sqrt(torch.pow(freq_radius[0], 2) + \
                                torch.pow(freq_radius[1], 2)  + \
                                torch.pow(freq_radius[2], 2))
    return freq_radius

def _normalised_cc_real_1d(tensor1: torch.Tensor, tensor2:torch.Tensor):
    correlation = torch.dot(tensor1, torch.conj(tensor2))
    return correlation / (torch.linalg.norm(tensor1) * torch.linalg.norm(tensor2))

def compute_rfsc(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """Computes the FSC between tensor1 and tensor2 which need not be square"""
    assert tensor1.shape == tensor2.shape
    original_shape = tensor1.shape
    freq_radius = make_rfreq_radius_new(torch.clone(torch.as_tensor(tensor1.shape)))
    tensor1 = torch.fft.rfftn(tensor1)
    tensor2 = torch.fft.rfftn(tensor2)
    sorted_frequencies_flat, sort_idx_flat = torch.sort(torch.flatten(freq_radius), descending = False)
    del(freq_radius)
    #spacing allows our spherical shell to encompass all data instead of the largest centered sphere, 
    #ie a spherical slice out of a thin rectanular prism
    bin_centers = torch.fft.rfftfreq(int(torch.min(torch.as_tensor(original_shape))))
    bin_interval = bin_centers[1]

    bin_centers = torch.cat([bin_centers, torch.arange(start = bin_centers[-1] + bin_interval,
                                                       end = np.sqrt(len(original_shape) * 0.25) + bin_interval,
                                                        step = bin_interval )])
    bin_centers = bin_centers.unfold(dimension=0, size=2, step=1)
    bin_centers = torch.mean(bin_centers, dim = 1)
    split_idx   = torch.searchsorted(sorted_frequencies_flat, bin_centers)
    shell_index = torch.tensor_split(sort_idx_flat, split_idx)[:-1] #exlude last one because of binning
    del(sorted_frequencies_flat, sort_idx_flat)
    tensor1 = torch.flatten(tensor1)
    tensor2 = torch.flatten(tensor2)

    fsc = torch.tensor([
    _normalised_cc_real_1d(tensor1[idx], tensor2[idx])
    for idx in shell_index
    ])

    return bin_centers, fsc