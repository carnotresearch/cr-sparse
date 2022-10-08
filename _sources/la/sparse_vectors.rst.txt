Sparse Vectors
========================


In this section we explore some useful properties of  :math:`\Sigma_k`, the set of  :math:`k`-sparse signals in standard basis
for  :math:`\mathbb{C}^n`.

We recall that

.. math::

    \Sigma_k  = \{ x \in \mathbb{C}^n : \| x \|_0 \leq k \}.


This set is a union of  :math:`\binom{n}{k}` subspaces of  :math:`\mathbb{C}^n` each of which
is is constructed by an index set  :math:`\Lambda \subset \{1, \dots, n \}` with  :math:`| \Lambda | = k` choosing
:math:`k` specific dimensions of  :math:`\mathbb{C}^n`. 

We first present some lemmas which connect the  :math:`l_1`,  :math:`l_2` and  :math:`l_{\infty}` norms of vectors
in  :math:`\Sigma_k`.

.. _lem:u_sigma_k_norms:

.. lemma::


    
    Suppose  :math:`u \in \Sigma_k`.  Then
    
    .. math::
    
          \frac{\| u\|_1}{\sqrt{k}} \leq \| u \|_2 \leq \sqrt{k} \| u \|_{\infty}.
    



.. proof::

   We can write  :math:`l_1` norm as
    
    .. math::
    
        \| u \|_1 = \langle u, \sgn (u) \rangle.
    
    
    By Cauchy-Schwartz inequality we have
    
    .. math::
    
        \langle u, \sgn (u) \rangle \leq  \| u \|_2  \| \sgn (u) \|_2 
     
    
    Since  :math:`u \in \Sigma_k`,  :math:`\sgn(u)` can have at most  :math:`k` non-zero values each with magnitude 1.
    Thus, we have
    
    .. math::
    
        \| \sgn (u) \|_2^2 \leq k \implies \| \sgn (u) \|_2 \leq \sqrt{k}
    
    
    Thus we get the lower bound
    
    .. math::
    
        \| u \|_1 \leq \| u \|_2 \sqrt{k}
        \implies \frac{\| u \|_1}{\sqrt{k}} \leq \| u \|_2.
    
    
    Now  :math:`| u_i | \leq \max(| u_i |) = \| u \|_{\infty}`. So we have
      
    .. math::
    
          \| u \|_2^2 = \sum_{i= 1}^{n} | u_i |^2 \leq  k \| u \|_{\infty}^2
    
    since there are only  :math:`k` non-zero terms in the expansion of  :math:`\| u \|_2^2`.
    
    This establishes the upper bound:
    
    .. math::
    
          \| u \|_2 \leq \sqrt{k} \| u \|_{\infty}
    


