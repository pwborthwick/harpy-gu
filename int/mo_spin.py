from __future__ import division
import numpy as np

def orbital_transform(scf, mode, tensor, c=None):
    #do ao, mo and spin orbital transformation

    def ao_transform_mo(c, tensor):
        #transform atomic orbital to molecuular

        tensor_type = tensor.ndim 

        if tensor_type == 1: transformed_tensor = tensor

        if tensor_type == 2:
            transformed_tensor = np.einsum('pQ,pP->PQ', np.einsum('pq,qQ->pQ', tensor, c, optimize=True), c, optimize=True)

        if tensor_type == 4:
            transformed_tensor = np.einsum('pQRS,pP->PQRS', np.einsum('pqRS,qQ->pQRS', np.einsum('pqrS,rR->pqRS', np.einsum('pqrs,sS->pqrS',  
                                                tensor, c, optimize=True), c, optimize=True), c, optimize=True), c, optimize=True)

        return transformed_tensor

    def transform_spin(tensor):
        #transform to spin - < || > in physicist's notation
        #m - molecular basis, s - spin basis, b returns the block mo conversion tensor

        tensor_type = tensor.ndim

        if tensor_type == 1:
            transformed_tensor = np.kron(tensor, np.ones(2))


        if tensor_type == 2:
            transformed_tensor = np.kron(np.eye(2), tensor)

        if tensor_type == 4:
            spin_block = np.kron(np.eye(2), np.kron(np.eye(2), tensor).transpose())
            transformed_tensor = spin_block.transpose(0,2,1,3) - spin_block.transpose(0,2,3,1)

        return transformed_tensor


    if type(c) != np.ndarray: c = scf.get('c')

    if mode == 'm': return ao_transform_mo(c, tensor)
    if mode == 's': return transform_spin(tensor)
    if mode in ['m+s', 's+m', 'b']:
        tensor = transform_spin(tensor)

        c_block = np.block([ [c, np.zeros_like(c)],
                             [np.zeros_like(c), c]])
        e = scf.get('e')
        eps = np.concatenate((e,e), axis=0)
        c = c_block[:, eps.argsort()]

        #return just the conversion block
        if mode == 'b': return c             

        return ao_transform_mo(c, tensor)


def orbital_deltas(scf, level, mo='s', e=None):
    #get orbital difference tensor, s-spin, x-spatial

    nocc = sum(scf.mol.nele)//2 if mo == 'x' else sum(scf.mol.nele)
    eps  = scf.get('e')         if mo == 'x' else orbital_transform(scf, 'm+s', scf.get('e'))

    #get orbital slices
    o, v, n = slice(None, nocc), slice(nocc, None), np.newaxis

    #use supplied e
    if type(e) == np.ndarray: eps = e

    #deltas
    ds = (eps[o, n] - eps[n, v] )
    dd = (eps[o, n, n, n] + eps[n, o, n, n] - eps[n, n, v, n] - eps[n, n, n, v] )
    dt = (eps[o, n, n, n, n, n] + eps[n, o, n, n, n, n] + eps[n, n, o, n, n, n]- 
          eps[n, n, n, v, n, n] - eps[n, n, n, n, v, n] - eps[n, n, n, n, n, v] )

    return [ds, dd, dt][:level]

def spin_to_spatial(tensor, type='x'):
    #transform a spin ndim==2 tensor to spatial

    if tensor.ndim != 2:
        exit('Not implemented')

    if type == 'x':
        return (tensor[::2,::2] + tensor[1::2,1::2] + tensor[::2,1::2] + tensor[1::2,::2]) 

    return (tensor[::2,::2] + tensor[1::2,1::2] + tensor[::2,::2].transpose(1,0) + tensor[1::2,1::2].transpose(1,0)) * 0.5
