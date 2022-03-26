#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np


# In[293]:


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def sinkhorn_iterations(Z: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor, iters: int) -> torch.Tensor:
    u, v = torch.ones_like(mu), torch.ones_like(nu)
    for _ in range(iters):
        u = mu / torch.einsum('bjk,bk->bj', [Z, v])
        v = nu / torch.einsum('bkj,bk->bj', [Z, u])
    return torch.einsum('bk,bkj,bj->bjk', [u, Z, v])


# In[ ]:





# In[294]:


def sinkhorn(C, a, b, eps=1e-2, n_iter=50, log_space=True):
    """
    Args:
        a: tensor, normalized, note: no zero elements
        b: tensor, normalized, note: no zero elements
        C: cost Matrix [batch, n_dim, n_dim], note: no zero elements
    """
    P = torch.exp(-C/eps)
    if log_space:
        log_a = a.log()
        log_b = b.log()
        log_P = P.log()
    
        # solve the P
        log_P = log_sinkhorn_iterations(log_P, log_a, log_b, n_iter)
    
        P = torch.exp(log_P)
    else:
        P = sinkhorn_iterations(P, a, b, n_iter)
    
    return torch.sum(C * P), P


# In[295]:


a = torch.tensor([1., 2., 3., 8.]).exp()
a = a / torch.sum(a)


# In[ ]:





# In[309]:


b = torch.tensor([1., 1., 3, 4.]).exp()
b = b / torch.sum(b)


# In[310]:


a, b


# In[311]:


C = -torch.eye(4) + 1e-6
C = C.exp()/torch.sum(C.exp())


# In[312]:


C


# In[313]:


a = a.reshape(1, 4)
b = b.reshape(1, 4)
C = C.reshape(1, 4, 4)
loss, P = sinkhorn(C, a, b, eps=1e-2)
print(loss)
print(P)


# In[314]:


loss, P = sinkhorn(C, a, b, log_space=False)
print(loss)
print(P)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




