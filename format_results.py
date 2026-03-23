"""
THE FOLLOWING code loads the optimal APNs and outputs a nicely formatted table in LaTex code for the paper.
Should be run after classify_GR42_APNs.py.
"""

import numpy as np
from utils import ddt_func_list


optimals = np.load("./apns-R4-4-ordered.npy")


# --------- Printing the table with functions --------- #
for i,f in enumerate(optimals):
    func_string = ""
    for el in f:
        if len(str(int(el))) == 1:
            func_string += f"\\phantom{{0}}{el}, "
        else:
            func_string += f"{str(el)}, "
        
    func_string = f"\\texttt{{{func_string[:-2]}}}"

    print(f"$\\GR_{{{i}}}$ & {func_string} \\\\ \\hline")


print()
print()

# --------- Printing the table spectrums --------- #

spectrums = set()

for f in optimals:
    spec3 = np.bincount(ddt_func_list( f,4 ).flatten())[:3]
    spectrums.add( tuple( spec3 ))

spectrumsList = list(sorted(spectrums))

for i,spec in enumerate(spectrumsList):
    functions = []
    for j,f in enumerate(optimals):
        fSpec = np.bincount(ddt_func_list( f,4 ).flatten())[:3]
        if tuple(fSpec) == spec:
            functions.append(j)
    print(f"$S_{{{i}}}$ & ${spec[0]}$ & ${spec[1]}$ & ${spec[2]}$ & ${", ".join( [f"\\GR_{{{ el }}}" for el in functions ] )}$ \\\\ \\hline")