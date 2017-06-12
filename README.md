# ph_analysis
Analysis tools for phonons

This package includes:
- Analysis tools for force constants

FCSymmetrizer
=============

The appropriate input is not "primitive_matrix" but "atoms_ideal".
This is because, e.g., hcp (A3) and Bh structures have the same
atomic positions in the primitive cell but still are different.