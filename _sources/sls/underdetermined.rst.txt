Underdetermined Linear Systems
=================================


The discussion in this section is largely based on chapter 1 of 
:cite:`elad2010sparse`.

Consider a matrix :math:`\Phi \in \CC^{M \times N}` with :math:`M < N`. 

Define an under-determined system of linear equations:

.. math::

  \Phi x = y

where :math:`y \in \CC^M` is known and :math:`x \in \CC^N` is unknown. 

This system has :math:`N` unknowns and
:math:`M` linear equations. 
There are more unknowns than equations.

Let the columns of :math:`\Phi` be given by :math:`\phi_1, \phi_2, \dots, \phi_N`.

Column space of :math:`\Phi` (vector space spanned by all columns of :math:`\Phi`)  is denoted by :math:`\ColSpace(\Phi)`
i.e.

.. math::

  \ColSpace(\Phi) = \sum_{i=1}^{N} c_i \phi_i, \quad c_i \in \CC.


We know that :math:`\ColSpace(\Phi) \subset \CC^M`. 

Clearly :math:`\Phi x \in \ColSpace(\Phi)` for every :math:`x \in \CC^N`.  Thus if :math:`y \notin \ColSpace(\Phi)` then we have no solution. But, if :math:`y \in \ColSpace(\Phi)` then we have infinite number of solutions.

Let :math:`\NullSpace(\Phi)` represent the null space of :math:`\Phi` given by 

.. math::

  \NullSpace(\Phi) = \{ x \in \CC^N : \Phi x = 0\}.


Let :math:`\widehat{x}` be a solution of :math:`y = \Phi x`. And let :math:`z \in \NullSpace(\Phi)`. Then 

.. math::

  \Phi (\widehat{x} + z) = \Phi \widehat{x} + \Phi z = y + 0  = y.

Thus the set :math:`\widehat{x} + \NullSpace(\Phi)` forms the complete set of infinite solutions to the
problem :math:`y = \Phi x` where

.. math::

  \widehat{x} + \NullSpace(\Phi) = \{\widehat{x} + z \quad \Forall z \in \NullSpace(\Phi)\}.

Regularization
------------------------
One way to pick a specific solution from the set of all possible solutions is to introduce **regularization**. 

.. index:: Regularization

We define a cost function :math:`J(x) : \CC^N \to \RR` which defines the **desirability** of a given solution :math:`x` out
of infinitely possible solutions. The higher the cost, lower is the desirability of
the solution.

.. index:: Desirability

Thus the goal of the optimization problem is to find a desired :math:`x` with minimum possible cost.

We can write this optimization problem as
  
.. math::

  \begin{aligned}
    & \underset{x}{\text{minimize}} 
    & &  J(x) \\
    & \text{subject to}
    & &  y = \Phi x.
  \end{aligned}



If :math:`J(x)` is convex, then its possible to find a global minimum cost solution over the solution set.

If :math:`J(x)` is not convex, then it may not be possible to find a global minimum, we may have to
settle with a local minimum. 

A variety of such cost function based criteria can be considered. 

:math:`l_2` Regularization
--------------------------------

One of the most common criteria is to choose a solution with the smallest :math:`l_2` norm.

The problem can then be reformulated as an optimization problem 
  
.. math::

  \begin{aligned}
    & \underset{x}{\text{minimize}} 
    & &  \| x \|_2 \\
    & \text{subject to}
    & &  y = \Phi x.
  \end{aligned}


In fact minimizing :math:`\| x \|_2` is same as minimizing its square :math:`\| x \|_2^2 = x^H x`.

So an equivalent formulation is 

  
.. math::

  \begin{aligned}
    & \underset{x}{\text{minimize}} 
    & &  x^H x \\
    & \text{subject to}
    & &  y = \Phi x.
  \end{aligned}



A formal solution to :math:`l_2` norm minimization problem can be easily obtained using
Lagrange multipliers.

We define the Lagrangian
  
.. math::

  \mathcal{L}(x) = \|x\|_2^2 + \lambda^H (\Phi x  - y)

with :math:`\lambda \in \CC^M` being the Lagrange multipliers for the (equality) constraint set.

Differentiating :math:`\mathcal{L}(x)` w.r.t. :math:`x` we get
  
.. math::

  \frac{\partial \mathcal{L}(x)} {\partial x} = 2 x + \Phi^H \lambda.


By equating the derivative to :math:`0` we obtain the optimal value of :math:`x` as
  
.. math::

  x^* = - \frac{1}{2} \Phi^H \lambda.
  \label{eq:ssm:underdetermined_l2_optimal_value_expression_1}


Plugging this solution back into the constraint :math:`\Phi x= y` gives us
  
.. math::

  \Phi x^* = - \frac{1}{2} (\Phi \Phi^H) \lambda= y\implies \lambda = -2(\Phi \Phi^H)^{-1} y.


In above we are implicitly assuming that :math:`\Phi` is a full rank matrix thus, :math:`\Phi \Phi^H` is invertible
and positive definite.

Putting :math:`\lambda` back in above we obtain
the well known closed form least squares solution using pseudo-inverse solution
  
.. math::

  x^* = \Phi^H (\Phi \Phi^H)^{-1} y = \Phi^{\dag} y.

Convexity
------------------
Convex optimization
problems have a unique feature that it is possible to find the global optimal solution if
such a solution exists. 

The solution space  :math:`\Omega = \{x : \Phi x = y\}` is convex.
Thus the feasible set of solutions for the optimization problem
is also convex. All it remains is to make sure that we choose a cost function
:math:`J(x)` which happens to be convex. This will ensure that a global minimum can be found through
convex optimization techniques. Moreover, if :math:`J(x)` is strictly convex, then it is guaranteed
that the global minimum solution is *unique*. Thus even though, we may not have
a nice looking closed form expression for the solution of a strictly convex cost function minimization problem,
the guarantee of the existence and uniqueness of solution as well as well developed algorithms
for solving the problem make it very appealing to choose cost functions which are convex.

We recall that all :math:`l_p` norms with :math:`p \geq 1` are convex functions.
In particular :math:`l_{\infty}` and :math:`l_1` norms are very interesting and popular where
  
.. math::

  l_{\infty}(x) = \max(|x_i|), \, 1 \leq i \leq N

and
  
.. math::

  l_1(x) = \sum_{i=1}^{N} |x_i|.


In the following section we will attempt to find a unique solution to our 
optimization problem using :math:`l_1` norm.

:math:`l_1` Regularization
-----------------------------------

In this section we will restrict our attention to the
Euclidean space case where :math:`x \in \RR^N`,
:math:`\Phi \in \RR^{M \times N}` and :math:`y \in \RR^M`.

We choose our cost function :math:`J(x) = l_1(x)`.

The cost minimization problem can be reformulated as
  
.. math::

  \begin{aligned}
    & \underset{x}{\text{minimize}} 
    & &  \| x \|_1 \\
    & \text{subject to}
    & &  \Phi x = y.
  \end{aligned}



It's time to have a closer look at our cost function :math:`J(x) = \|x \|_1`. This function
is convex yet not strictly convex. 

For the :math:`l_1` norm minimization problem since :math:`J(x)` is not strictly convex,
hence a unique solution may not be guaranteed. In specific cases, there may be
infinitely many solutions. Yet what we can claim is

* these solutions are gathered in a set that is bounded and convex, and
* among these solutions, there exists at least one solution with at most
  :math:`M` non-zeros (as the number of constraints in :math:`\Phi x = y`).


.. theorem::

  Let :math:`S` denote the solution set of :math:`l_1` norm minimization problem.
  :math:`S`
  contains at least one solution :math:`\widehat{x}` with
  :math:`\| \widehat{x} \|_0 = M`.

See :cite:`elad2010sparse` for proof.


We thus note that :math:`l_1` norm has a tendency to prefer sparse solutions. This is a
well known and fundamental property of linear programming.

:math:`l_1` norm minimization problem as a linear programming problem
------------------------------------------------------------------------

We now show that :math:`l_1` norm minimization problem in :math:`\RR^N` 
is in fact a linear programming problem.

Recalling the problem:
    
.. math::
  :label: l1_norm_minimization_problem

  \begin{aligned}
    & \underset{x \in \RR^N}{\text{minimize}} 
    & &  \| x \|_1 \\
    & \text{subject to}
    & &  y = \Phi x.
  \end{aligned}


Let us write :math:`x` as :math:`u  - v`  where :math:`u, v \in \RR^N` are both non-negative vectors such that
:math:`u` takes all positive entries in :math:`x` while :math:`v` takes all the negative entries in :math:`x`.

We note here that by definition
    
.. math::

  \supp(u) \cap \supp(v) = \EmptySet

i.e. support of :math:`u` and :math:`v` do not overlap.

We now construct a vector
    
.. math::

  z = \begin{bmatrix}
  u \\ v
  \end{bmatrix} \in \RR^{2N}.


We can now verify that
    
.. math::

  \| x \|_1 = \|u\|_1 + \| v \|_1 = 1^T z.


And 
    
.. math::

  \Phi x = \Phi (u - v) = \Phi u - \Phi v = 
  \begin{bmatrix}
  \Phi & -\Phi
  \end{bmatrix}
  \begin{bmatrix}
  u \\ v
  \end{bmatrix}
  = \begin{bmatrix}
  \Phi & -\Phi
  \end{bmatrix} z 

where  :math:`z \succeq 0`.

Hence the optimization problem :eq:`l1_norm_minimization_problem` can be recast as
    
.. math::
  :label: l1_norm_minimization_problem_as_lp

  \begin{aligned}
    & \underset{z \in \RR^{2N}}{\text{minimize}} 
    & &  1^T z \\
    & \text{subject to}
    & &  \begin{bmatrix} \Phi & -\Phi \end{bmatrix} z = y\\
    & \text{and}
    & & z \succeq 0.
  \end{aligned}


This optimization problem has the classic Linear Programming structure since the
objective function is affine as well as constraints are affine.

Let :math:`z^* =\begin{bmatrix} u^* \\ v^* \end{bmatrix}` be an optimal solution to the
problem :eq:`l1_norm_minimization_problem_as_lp`.  

In order to show that the two optimization problems are equivalent, we need
to verify that our assumption about the decomposition of :math:`x` into positive entries in :math:`u` 
and negative entries in :math:`v` is indeed satisfied by the optimal solution :math:`u^*` and :math:`v^*`.
i.e. support of :math:`u^*` and :math:`v^*` do not overlap.

Since :math:`z \succeq 0` hence :math:`\langle u^* , v^* \rangle  \geq 0`. If support of :math:`u^*` and :math:`v^*` 
don't overlap, then we  have :math:`\langle u^* , v^* \rangle = 0`. And if they overlap then
:math:`\langle u^* , v^* \rangle > 0`.

Now for the sake of contradiction, let us assume that support of :math:`u^*` and :math:`v^*` do overlap for 
the optimal solution :math:`z^*`.

Let :math:`k` be one of the indices at which both :math:`u_k \neq 0` and :math:`v_k \neq 0`. Since :math:`z \succeq 0`, 
hence :math:`u_k > 0` and :math:`v_k > 0`.

Without loss of generality let us assume that :math:`u_k > v_k > 0`.

In the equality constraint
    
.. math::

  \begin{bmatrix} \Phi & -\Phi \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = y


Both of these coefficients multiply the same column of :math:`\Phi` with opposite signs giving us a term
    
.. math::

  \phi_k (u_k - v_k). 


Now if we replace the two entries in :math:`z^*` by
    
.. math::

  u_k'  = u_k - v_k

and
    
.. math::

  v_k' = 0

to obtain an new vector :math:`z'`, 
we see that there is no impact in the equality constraint 
    
.. math::

  \begin{bmatrix} \Phi & -\Phi \end{bmatrix} z = y.

Also the positivity constraint
    
.. math::

  z \succeq 0

is satisfied. This means that :math:`z'` is a feasible solution.

On the other hand, the objective function :math:`1^T z` value reduces by :math:`2 v_k` for :math:`z'`. 
This contradicts our assumption that :math:`z^*` is the optimal solution.

Hence for the optimal solution of :eq:`l1_norm_minimization_problem_as_lp`
we have
    
.. math::

  \supp(u^*) \cap \supp(v^*) = \EmptySet

thus 
    
.. math::

  x^* = u^* - v^*

is indeed the desired solution for the optimization problem :eq:`l1_norm_minimization_problem`.


