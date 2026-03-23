"""
Since Numba does not work well with sage objects, we make a separate file with the util functions without the numba decorator.
"""
import numpy as np


def base_m_xor_numba(a, b, base):
    result = 0
    place = 1
    while a > 0 or b > 0:
        result += ((a % base + b % base) % base) * place
        a //= base
        b //= base
        place *= base
    return result


def base_m_xor_minus_numba(a, b, base):
    result = 0
    place = 1
    while a > 0 or b > 0:
        result += ((a % base - b % base) % base) * place
        a //= base
        b //= base
        place *= base
    return result

base_m_xor_numba = np.vectorize( base_m_xor_numba)
base_m_xor_minus_numba = np.vectorize( base_m_xor_minus_numba)


def isInvertible2dmatrix(M, base, p):
    # M is a 2x2 matrix of int64
    determinant = (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) % base
    return determinant % p != 0


def matmul2d( M,x,p,k,m ):
    """
    M: Z_{p^k} m x m matrix
    x: element in GR, uint16
    kan hende vi antar at p = 2
    """
    base = p**k

    assert M.shape == (m,m), "wrong dimensions on M"

    # deompose x into array:
    x0 = x % base
    x1 = (x // base) % base


    y0 = (M[0,0] * x0 + M[0,1] * x1) % base
    y1 = (M[1,0] * x0 + M[1,1] * x1) % base


    return y0 + base*y1


def int2vec( a,base,m ):
    a = int(a)
    v = np.empty( m,dtype=np.int64 )
    for i in range(m):
        v[i] = ( a//base**i ) % base 
    
    return v


def vec2int( v,base ):
    a = 0
    for i in range(len(v)):
        a += int(v[i])*base**i
    return a


def ddt_func_list(f, base):
    """
    f: lookup-table of lenght base^m 
    base: p^k in the Galois ring over which f is defined, where the ring is GR(p^k,m)
    returns the difference distribution table of f.
    """
    f = np.asarray( f,dtype=np.int64 )
    size = len(f)  # Length of the array, we assert it is power of 2
    assert (size & (size - 1)) == 0, f"Length of table must be a power of 2. Got size {size}"
    
    # Initialize the DDT as a zero matrix
    ddt = np.zeros( (size,size), dtype=np.int64 )
    
    # Compute DDT by iterating over all possible input differences (Delta x)
    for a in range(size):
        x = np.arange(size, dtype=np.int64 )
        x_prime = base_m_xor_numba(x, a, base)
        b = base_m_xor_minus_numba(f[x], f[x_prime], base)
        
        # Count occurrences of each output difference b
        ddt[a] = np.bincount( b,minlength=size )
    
    return ddt


def diff_spec_func_list( f,base ):
    ddt = ddt_func_list( f,base )
    spectrum = np.bincount( ddt.flatten() )

    return spectrum

