"""
This code takes as input a list of permutations, and classifies them up to CCZ-equialvence. Finally, it provides the classes.
Should be run after classify_GR42_APNs.py.
"""
from itertools import product
import numpy as np
from utils_sage import base_m_xor_numba, int2vec, isInvertible2dmatrix, vec2int, diff_spec_func_list
from time import perf_counter as pf


p = 2
k = 2
m = 2

base = p^k



def are_ccz_equivalent_4_2( f,g ):
    R = Integers(4)
    
    # all matrices in Z_4^2
    inv_mats = []
    mats = []
    for i in range( int(4**4) ):
        M = np.array( [[ i // 4**j % 4 for j in range(2)],[i // 4**j % 4 for j in range(2,4)]],np.uint16 )
        if isInvertible2dmatrix( M,base,p ):
            inv_mats.append(M)
        mats.append(M)

    I = np.array( [[1,0],[0,1]],np.uint16 )
    np_inv_mats = np.array( inv_mats,np.uint16 )
    np_mats = np.array( mats,np.uint16 )

    desiredGraphImage = np.zeros( (2,16),np.uint16 )
    xPermuted = np.zeros( 16,np.uint16 )

    H = np.empty( (16),np.uint16 )
    cnt = 0
    for A,B in product( np_mats,repeat=2 ):
        cnt += 1
        # Just check if the image of the x-coordinate of graph is bijective.
        domain = set()
        for x in range( 16 ):
            xVec = int2vec( x,base,m )
            xPrime = (np.dot( A,xVec.T ) + np.dot( B,int2vec(f[x],base,m ).T )) % base
            xPrimePacked = vec2int( xPrime,base )
            if xPrimePacked in domain:
                break
            domain.add( xPrimePacked )
            xPermuted[x] = xPrimePacked
        else:
            for a1Packed in range(16):
                
                a1vec = int2vec( a1Packed,base,m )
                desiredGraphImage[ 0,: ] = 0
                for x in range(16):
                    desiredGraphImage[0,x] = base_m_xor_numba( xPermuted[x],a1Packed,4 )
                # Compute the desired solution
                desiredGraphImage[1,:] = g[desiredGraphImage[0,:]]

                # We have a system of equations.
                # Know C 

                # x = [ C11 C12 C21 C22 D11 D12 D21 D22 a21 a22 ]

                # Two different types of rows in M:
                # First  kind C11 xi1 + C12 xi2 + D11 F(xi)1 + D12 F(xi)2 + a1 = yi'1 mod 4
                # Second kind C21 xi1 + C22 xi2 + D21 F(xi)1 + D22 F(xi)2 + a2 = yi'2 mod 4
                M = np.zeros( (32,10),np.uint16 )
                for i in range( 0,32,2 ):
                    
                    xi = i//2
                    xiVec = int2vec( xi,base,m )
                    fxi = f[xi]
                    fxiVec = int2vec( fxi,base,m )

                    M[i  ,0] = xiVec[0]
                    M[i  ,1] = xiVec[1]
                    M[i  ,4] = fxiVec[0]
                    M[i  ,5] = fxiVec[1]
                    M[i  ,8] = 1
                    
                    M[i+1,2] = xiVec[0]
                    M[i+1,3] = xiVec[1]
                    M[i+1,6] = fxiVec[0]
                    M[i+1,7] = fxiVec[1]
                    M[i+1,9] = 1

                M = Matrix(R,M)
                y = []
                for el in desiredGraphImage[1,:]:
                    y += list( int(el) for el in int2vec(el,base,m))
                y = vector(R,y)


                try:
                    sol = M.solve_right(y)

                    A = matrix(R,A)
                    B = matrix(R,B)
                    C = matrix(R,np.array( [[sol[0], sol[1]],[sol[2], sol[3]]], np.uint16 ))
                    D = matrix(R,np.array( [[sol[4], sol[5]],[sol[6], sol[7]]], np.uint16 ))
                    a2 = vector(R, sol[-2:])
                    MM = block_matrix(R, [[A,B],[C,D]] )


                    # deomnstrate that the graph is mapped correctly:
                    gMapped = np.zeros(16)
                    ok = True
                    for x in range(16):
                        xVec = vector(R,int2vec( x,base,m ))
                        yVec = vector(R,int2vec( f[x],base,m ))

                        graphVec = vector( R,list(int2vec( x,base,m )) + list( int2vec( f[x],base,m ) ) )

                        aVec = vector(R, list(a1vec) + list(a2))
                        
                        graphVecTransformed = MM * graphVec + aVec

                        new_x = vec2int(graphVecTransformed[:2],base)
                        new_y = vec2int(graphVecTransformed[2:],base)

                        if not g[new_x] == new_y:
                            ok = False, None
                            break
                        else:
                            return True, (MM,aVec)
                except ValueError:
                    pass

    return False,None




def classify_ccz( functions ):
    functions = list(functions)
    classes = []
    
    i = 0
    for i,f in enumerate(functions):
        placed = False
        f_spec = diff_spec_func_list(f,4)

        for cl in classes:
            if i in cl:
                placed = True
                break
        
        if placed:
            continue
        else:
            classes.append( [i] )

        for j in range( i+1,len(functions) ):
            g = functions[j]
            g_spec = diff_spec_func_list( g,4 )
            if np.all( f_spec == g_spec ):
                print(f" Checking {i} and {j} ")
                #equiv = bruteForceCCZ_parallell( f,g )
                equiv,witness = are_ccz_equivalent_4_2( f,g )
                # print(f"Permutations {i} and {j} are equivalent?")
                if equiv:
                    print(f"Equivalent under the following affine transformation:")
                    classes[-1].append(j)
                    print(witness[0])
                    print(witness[1])
                else:
                    print("Not equivalent!")

    print(f"Found {len(classes)} CCZ-classes:")
    print(classes)
    return classes


if __name__ == "__main__": 
    orderedOptimals = np.load( "./apns-R4-4-ordered.npy")
    orderedTuples = set( tuple(f) for f in orderedOptimals )

    classify_ccz( orderedOptimals )


