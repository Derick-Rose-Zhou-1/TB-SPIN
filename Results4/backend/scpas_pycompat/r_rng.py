from __future__ import annotations
import numpy as np

class RMT19937:
    """R's Mersenne-Twister initialization and 32-bit output stream."""
    N=624; M=397; MATRIX_A=0x9908B0DF; UPPER_MASK=0x80000000; LOWER_MASK=0x7FFFFFFF
    def __init__(self, seed:int): self.set_seed(seed)
    def set_seed(self, seed:int):
        seed &= 0xffffffff
        for _ in range(50): seed=(69069*seed+1)&0xffffffff
        state=[]
        for _ in range(625): seed=(69069*seed+1)&0xffffffff;state.append(seed)
        self.mti=state[0];self.mt=state[1:]
    def uint32(self):
        if self.mti>=self.N:
            mt=self.mt
            for kk in range(self.N-self.M):
                y=(mt[kk]&self.UPPER_MASK)|(mt[kk+1]&self.LOWER_MASK);mt[kk]=mt[kk+self.M]^(y>>1)^(self.MATRIX_A if y&1 else 0)
            for kk in range(self.N-self.M,self.N-1):
                y=(mt[kk]&self.UPPER_MASK)|(mt[kk+1]&self.LOWER_MASK);mt[kk]=mt[kk+(self.M-self.N)]^(y>>1)^(self.MATRIX_A if y&1 else 0)
            y=(mt[-1]&self.UPPER_MASK)|(mt[0]&self.LOWER_MASK);mt[-1]=mt[self.M-1]^(y>>1)^(self.MATRIX_A if y&1 else 0);self.mti=0
        y=self.mt[self.mti];self.mti+=1
        y^=y>>11;y^=(y<<7)&0x9D2C5680;y^=(y<<15)&0xEFC60000;y^=y>>18
        return y&0xffffffff
    def uniform(self):
        x=self.uint32()/4294967296.0
        if x<=0: return 0.5/4294967296.0
        if x>=1: return 1-0.5/4294967296.0
        return x

def r_sample_indices(n:int, seed:int):
    """R sample.int(n,n,replace=FALSE) for R >=3.6 default rejection sampler."""
    rng=RMT19937(seed);x=list(range(n));out=np.empty(n,dtype=np.int64)
    # R's sample.int default sample.kind='Rejection'. Fisher-Yates using R_unif_index.
    def bits(bits):
        v=0
        # Literal R rbits(): for (int n = 0; n <= bits; n += 16).
        for _ in range(bits//16+1): v=65536*v+int(rng.uniform()*65536)
        return v & ((1 << bits) - 1)
    def unif_index(nn):
        if nn<=0:return 0
        b=(nn-1).bit_length()
        while True:
            v=bits(b)
            if v<nn:return v
    for i in range(n):
        j=unif_index(n-i);out[i]=x[j];x[j]=x[n-i-1]
    return out

def permutation_indices(n:int, permutation_index:int):
    return r_sample_indices(n,1234+int(permutation_index))

def cv_foldid(n:int,nfolds:int,seed:int=123):
    base=np.resize(np.arange(1,nfolds+1,dtype=np.int64),n)
    return base[r_sample_indices(n,seed)]
