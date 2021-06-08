Sparsifying Dictionaries and Sensing Matrices
===================================================


.. currentmodule:: cr.sparse.dict


Functions for constructing sparsying dictionaries and sensing matrices
--------------------------------------------------------------------------


.. autosummary::
  :toctree: _autosummary


    gaussian_mtx
    rademacher_mtx
    random_onb
    hadamard
    hadamard_basis
    dirac_hadamard_basis
    dct_basis
    dirac_dct_basis
    dirac_hadamard_dct_basis
    fourier_basis



Dictionary properties
-------------------------


.. autosummary::
  :toctree: _autosummary

    gram
    coherence_with_index
    coherence
    frame_bounds
    upper_frame_bound
    lower_frame_bound
    babel


Dictionary comparison
----------------------------

These functions are useful for comparing dictionaries
during the dictionary learning process. 

.. autosummary::
  :toctree: _autosummary

    mutual_coherence_with_index
    mutual_coherence
    matching_atoms_ratio