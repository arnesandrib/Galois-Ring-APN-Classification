"""
The following code exhaustively searches all permutations of GR(4,2) with specified fixed points, and stores all APN permutations in a file.
"""
import numpy as np
from math import log2
from utils import base_m_xor_minus_numba, base_m_xor_numba, isAPN_func_list_precomputations
from numba import njit,int64
from numba.typed import List
from math import factorial


p = 2
k = 2
n = 2
base = p**k

nElements = base**n

def computeSumTable( p,k,n ):
    """
    Returns the table of sums in GR(p^k,n)
    """
    sums = np.empty( (p**(k*n),p**(k*n)),np.int64 )
    for x in range( p**(k*n) ):
        for y in range( p**(k*n) ):
            sums[x,y] = base_m_xor_numba( int64(x),int64(y),int64(base) )
    return sums

def computeDiffTable( p,k,n ):
    """
    Returns the table of differences in GR(p^k,n)
    """
    diffs = np.empty( (p**(k*n),p**(k*n)),np.int64 )
    for x in range( p**(k*n) ):
        for y in range( p**(k*n) ):
            diffs[x,y] = base_m_xor_minus_numba( int64(x),int64(y),int64(base) )
    return diffs

@njit
def update_permutation(permutation):
    """
    Taking an array elements in some order, and updates it to the next lexicographic element.
    """

    a = permutation
    n = len(a)
    i = n - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    
    # If no such i, we have the last permutation
    
    if not i >= 0: return False
    
    # Find largest j > i such that a[j] > a[i]
    j = n - 1
    while a[j] <= a[i]:
        j -= 1
    
    # Swap a[i], a[j]
    temp = a[i]
    a[i] = a[j]
    a[j] = temp
    
    # Reverse suffix a[i+1:]
    left = i + 1
    right = n - 1
    while left < right:
        temp = a[left]
        a[left] = a[right]
        a[right] = temp
        left += 1
        right -= 1
    
    return True

@njit
def search( p,k,n,sums,diffs,initPerm,numCalls ):
    base = p**k
    nElements = base**n

    nApns = 0

    APNs = List.empty_list(int64[:])

    # We fix 0, 1 and 4. Then we must search through approximately 2^32 functions.
    perm = np.arange( nElements )
    indices = np.array( [2,3,5,6,7,8,9,10,11,12,13,14,15],np.int64 )
    cnt = 0
    zeroTable  = np.zeros( (nElements,nElements ),dtype=np.int64 )

    perm13 = initPerm
    for _ in range( numCalls ):
        perm[ indices ] = perm13
        cnt += 1

        # Set the table to zero again.
        zeroTable[ : ] = 0
        isAPN = isAPN_func_list_precomputations( perm,zeroTable,sums,diffs )

        if isAPN:
            nApns += 1
            APNs.append( perm.copy() )
        
        if not update_permutation( perm13 ):
            break

    print(perm13)
    return APNs,perm13.copy()


def searchHandler(p,k,n):
    """
    Search handler for searching over APN permutations in GR(4,2). Do this to chunk the search and avoid Numba crashing.
    """
    sums = computeSumTable( p,k,n )
    diffs = computeDiffTable( p,k,n )

    initPerm = np.array( [2,3,5,6,7,8,9,10,11,12,13,14,15],np.int64 ) # increasing order, all but 0,1,4
    
    # We compute in chunks because @njit sometimes crashes when working too long at a time.
    chunkSize = factorial(9)
    numCalls = factorial(13) // chunkSize
    print(f"Number of calls iterating {chunkSize} = 2^{log2(chunkSize):.2f} functions to be made: {numCalls} = 2^{log2(numCalls):.2f}")

    allApns = []

    for i in range(numCalls):
        APNs,nextInit = search( p,k,n,sums,diffs,initPerm,chunkSize )
        allApns += APNs
        print("Next init:", nextInit)
        print(f"Detected apns: {len(allApns)}")
        print()

    assert np.all(np.array([15,14,13,12,11,10,9,8,7,6,5,3,2], dtype=np.int64) - nextInit == 0), "Did not cover all permutations."
    
    allApnsNumpy = np.array(allApns, np.int64)
    np.save( "./all-42-apns.npy",allApnsNumpy ) # Save all optimals with 0,1,4 fixed to file.

    with open("./all-42-apns.txt", "w") as file:
        for el in allApns:
            file.write( str( [ int(ell) for ell in el] ) + ",\n")

    print(f"Terminated with success, finding {len(allApns)} APN permutations in GR(4,2) with points 0,1,4 fixed.")
    

if __name__ == "__main__":
    searchHandler(p,k,n)
