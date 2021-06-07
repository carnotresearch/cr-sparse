from cr.sparse.pursuit.eval import SuccessRates


evaluation = SuccessRates(
    M = 200,
    N = 1000,
    Ks = range(2, 120+1),
    num_dict_trials = 100,
    num_signal_trials = 5
)

# Add solvers
from cr.sparse.pursuit import iht
from cr.sparse.pursuit import htp
from cr.sparse.pursuit import sp
from cr.sparse.pursuit import cosamp

evaluation.add_solver('IHT', iht.solve_jit)
evaluation.add_solver('HTP', htp.solve_jit)
evaluation.add_solver('SP', sp.solve_jit)
evaluation.add_solver('CoSaMP', cosamp.solve_jit)

# Run evaluation
evaluation()

# Save results
evaluation.save()
