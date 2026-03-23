# Galois-Ring-APN-Classification

The code provides efficient arithmetic over Galois rings using Python, NumPy and Numba.

The code also provides an implementation of a search for APN permutations over the Galois ring $\mathrm{GR}(4,2) \cong \mathbb{Z}_4 \times  \mathbb{Z}_4$. To reproduce the search, run the files in the following order
- GR42_APN_exhaustive.py: Exhaustively searches through all permutations of $\mathrm{GR}(4,2)$ with three specified fixed points. Produces the files "all-42-apns.npy" and "all-42-apns.txt", which contain the resulting functions, stored as tables. 
- classify_GR42_APNs.py: Extracts from the list of APN permutations a list of affine inequivalent representatives. All the APN permutations are affine equivalent to exacly one such representative. The program also outputs the differential spectrum of every class.
- format_results.py: Formats into neatly formatted LaTex-tables
- GR42_CCZ.sage: Sorts all functions in some list into their respective CCZ-classes. Written in sage to utilize the equation solver over modular rings, in particular $\mathbb{Z}_4$.
