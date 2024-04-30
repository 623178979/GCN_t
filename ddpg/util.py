import torch

def gather_nd(params, indices):
    '''
    4D example
    params: tensor shaped [n_1, n_2, n_3, n_4] --> 4 dimensional
    indices: tensor shaped [m_1, m_2, m_3, m_4, 4] --> multidimensional list of 4D indices
    
    returns: tensor shaped [m_1, m_2, m_3, m_4]
    
    ND_example
    params: tensor shaped [n_1, ..., n_p] --> d-dimensional tensor
    indices: tensor shaped [m_1, ..., m_i, d] --> multidimensional list of d-dimensional indices
    
    returns: tensor shaped [m_1, ..., m_1]
    '''

    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1) # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1
    
    for i in range(ndim)[::-1]:
        idx += indices[i] * m 
        m *= params.size(i)
    out = torch.take(params, idx)
    return out.view(out_shape)