# Galois-Ring-APN-Classification

Provides efficient arithmetic over Galois rings using Python, NumPy and Numba.

Also provides an implementation of a search for APN permutations over the Galois ring GR(4,2) \cong Z_4 \times Z_4. Reproduce the search, run the files in the following order
- GR42_APN_exhaustive.py: Exhaustively searches through all permutations of GR(4,2) with 3 specified fixed points. Produces the files "all-42-apns.npy" and "all-42-apns.npy", which contain the resulting functions, stored as tables. 
- classify_GR42_APNs.py: Extracts from the list of APN permutations a list of affine inequivalent representatives. All the APN permutations are affine equivalent to exacly one such representative. The program also outputs the differential spectrum of every class.
- format_results.py: Formats into neatly formatted LaTex-tables
- GR42_CCZ.sage: Sorts all functions in some list into their respective CCZ-classes. Written in sage to utilize the equation solver over modular rings, in particular Z_4.
