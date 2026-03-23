from numba import njit, vectorize, int64
import numpy as np

_DTYPE = int64

@vectorize([_DTYPE(_DTYPE,_DTYPE,_DTYPE)],cache=True)
def base_m_xor_numba(a, b, base):
    """
    Given elements a,b of Z_{base}^m as packed bits (integer representation). Returns a + b represented as integer.
    """
    result = 0
    place = 1
    while a > 0 or b > 0:
        result += ((a % base + b % base) % base) * place
        a //= base
        b //= base
        place *= base
    return result

@vectorize([_DTYPE(_DTYPE,_DTYPE,_DTYPE)],cache=True)
def base_m_xor_minus_numba(a, b, base):
    """
    Given elements a,b of Z_{base}^m as packed bits (integer representation). Returns a - b represented as integer.
    """
    result = 0
    place = 1
    while a > 0 or b > 0:
        result += ((a % base - b % base) % base) * place
        a //= base
        b //= base
        place *= base
    return result


@njit(cache=True)
def precomposeFuncAffine2d( F,p,k,m,M,a ):
    """
    Given function F in GR(2^k,2) as lookup-table, return the function G(x) = F( Mx + a ) as lookup-table
    """
    assert m == 2
    FPrime = np.zeros( len(F),np.uint16 )
    base = p**(k)

    for x in range(base**m):
        xPrime = matmul2d( M,x,p,k,m )
        xPPrime = base_m_xor_numba( xPrime,a,base )
        FPrime[x] = F[xPPrime]

    return FPrime

@njit(cache=True)
def postcomposeFuncAffine2d( F,p,k,m,M,a ):
    """
    Given function F in GR(2^k,2) as lookup-table, return the function G(x) = M F(x) + a as lookup-table
    """
    assert m == 2
    FPrime = np.zeros( len(F),np.uint16 )
    base = p**(k)

    for x in range(base**m):
        y = F[x]
        yPrime = matmul2d( M,y,p,k,m )
        yPPrime = base_m_xor_numba( yPrime,a, base )
        FPrime[x] = yPPrime

    return FPrime


@njit(cache=True)
def isInvertible2dMatrix( M,base,p ):
    """
    Decides if the 2x2 matrix M is invertible 
    """
    m = 2
    assert M.shape == (m,m), "wrong dimensions on M"

    determinant = (M[0,0] * M[1,1] - M[0,1]*M[1,0]) % base
    return determinant % p != 0


@njit(cache=True)
def matmul2d( M,x,p,k,m ):
    """
    M: Z_{p^k} m x m matrix
    x: element in GR, uint16
    p,k,m: parameters of the galois ring GR(p^k,m)
    """
    base = p**k
    #M = np.array( [[1,0],[1,1]],np.uint16 )

    assert M.shape == (m,m), "wrong dimensions on M"

    # deompose x into array:

    x0 = x % base
    x1 = (x // base) % base


    y0 = (M[0,0] * x0 + M[0,1] * x1) % base
    y1 = (M[1,0] * x0 + M[1,1] * x1) % base


    return y0 + base*y1

@njit(cache=True)
def ddt_func_list(f, base):
    """
    f: lookup-table of lenght base^m 
    base: p^k in the Galois ring over which f is defined, where the ring is GR(p^k,m)
    returns the difference distribution table of f.
    """
    f = np.asarray( f,dtype=_DTYPE )
    size = len(f)  # Length of the array, we assert it is power of 2
    assert (size & (size - 1)) == 0, f"Length of table must be a power of 2. Got size {size}"
    
    # Initialize the DDT as a zero matrix
    ddt = np.zeros( (size,size), dtype=_DTYPE )
    
    # Compute DDT by iterating over all possible input differences (Delta x)
    for a in range(size):
        x = np.arange(size, dtype=_DTYPE)
        x_prime = base_m_xor_numba(x, a, base)
        b = base_m_xor_minus_numba(f[x], f[x_prime], base)
        
        # Count occurrences of each output difference b
        ddt[a] = np.bincount( b,minlength=size )
    
    return ddt

@njit(cache=True)
def max_ddt(ddt):
    """
    Given a difference distribution table, returns the maximum value not in the first row.
    """
    max_val = _DTYPE(0)  # Start with the lowest possible value
    for i in range(1, ddt.shape[0]):  # Skipping row 0
        row_max = _DTYPE(0)
        for j in range(ddt.shape[1]):  # Iterate over columns
            if ddt[i, j] > row_max:
                row_max = ddt[i, j]  # Find max in row
        
        if row_max > max_val:
            max_val = row_max  # Find overall max

    return max_val

@njit(cache=True)
def diff_unif_func_list(f, base):
    """
    f: lookup-table of lenght base^m 
    base: p^k in the Galois ring over which f is defined, where the ring is GR(p^k,m)
    Returns the differential uniformity of f.
    """
    size = len(f)
    assert (size & (size - 1)) == 0, "Length of f must be a power of 2"
    ddt = ddt_func_list(f, base)
    return max_ddt(ddt)

@njit(cache=True)
def isAPN_func_list_precomputations( f,zero_table,sums,diffs ):
    """
    Determines if a function f is APN, given an empty DDT table and the addition/subtraction table of the function.
    """
    for a in range( 1,len(f) ):
        for x in range( len( f ) ):
            b = diffs[ f[ sums[x,a] ],f[ x ] ]
            zero_table[a,b] += 1
            if zero_table[a,b] > 2:
                return False
    return True


@vectorize([_DTYPE(_DTYPE,_DTYPE)],cache=True)
def negative(a,base):
    """
    Returns the additive inverse of a, where a is an element in GR(p^k,m), base = p^k, and a is represented as an integer.
    """
    return base_m_xor_minus_numba( _DTYPE(0),a,base )
