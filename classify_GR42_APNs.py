"""
Given a list of permutations, this code extracts representatives of affine equivalence classes, and provides their differential spectra.
Should be run after GR42_APN_exhaustive.py.
"""

import numpy as np
from utils import postcomposeFuncAffine2d, precomposeFuncAffine2d, matmul2d, isInvertible2dMatrix, ddt_func_list
from numba import njit,uint16
from numba.typed import List

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=np.inf)

p = 2
k = 2
m = 2
base = p**k

nElements = base**m

apns = np.load("./all-42-apns.npy")
fixedPoints = np.array( [0,1,4],np.uint16 )

# Compute all 2x2 invertible matrices over Z_{4}^2:
inv_mats = []
for i in range( 4**4 ):
    M = np.array( [[ i // 4**j % 4 for j in range(2)],[i // 4**j % 4 for j in range(2,4)]],np.uint16 )
    if isInvertible2dMatrix( M,base,p ):
        inv_mats.append(M)

np_inv_mats = np.array( inv_mats,np.uint16 )
print(f"Found {len(np_inv_mats)} invertible matrices.")


@njit(cache=True)
def checkFuncEquivalence( F,np_inv_mats,fixedPoints ):
    equivalents = List.empty_list(uint16[:])
    for A in np_inv_mats:
        for a in range(nElements):
            for B in np_inv_mats:
                for b in range(nElements):
                    FPrime = precomposeFuncAffine2d( F,p,k,m, A,a )
                    FPPrime = postcomposeFuncAffine2d( FPrime,p,k,m,B,b )
                    for fp in fixedPoints:
                        if FPPrime[fp] != fp: break
                    else:
                        equivalents.append(FPPrime.copy())

    return equivalents


# Creating dicionary of all imported APNs.
apnsDict = dict()
for el in apns:
    apnsDict[ tuple(el) ] = el

# List to hold final representatives of equivalence classes
apnsRepresentatives = []

# Go through every element of apnsDict 
while len(apnsDict) > 0:
    # Let the initial element of the dictionary be a representative.
    key = next(iter(apnsDict))
    apn = apnsDict.pop(key) 
    apnsRepresentatives.append(apn)
    
    # Generate all equivalent APNs with the same fixed points.
    equivalents = checkFuncEquivalence( apn,np_inv_mats,fixedPoints )

    # Remove from dictionary all elements equivalent to current representative.
    for el in equivalents:
        if tuple(el) in apnsDict:
            del apnsDict[ tuple(el) ]


# Save an ordered list of representatives of APN permutations.
np.save("./apns-R4-4-ordered.npy", apnsRepresentatives)

print()
print(f"Representatives:")
print(apnsRepresentatives)
print(len(apnsRepresentatives))

# Classify the spectrums.
lowest = p**(k*m)
highest = 0
nOptimals = 0
num2s = []
spectrums = set()

for apn in apns: # Recall apns only consist of one permutation for each affine equivalence class.
    DDT = ddt_func_list( apn,base )
    spectrum = np.bincount( DDT.flatten() )
    num2s.append(spectrum[2])
    lowest = min( lowest,spectrum[2] )
    highest = max( highest,spectrum[2] )
    spectrums.add(tuple(spectrum))
    if spectrum[2] == 64: nOptimals += 1


print("num spectrums:",len(spectrums))
for spectrum in spectrums:
    print( np.array( spectrum,np.uint16 ) )