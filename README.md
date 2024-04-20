# self_similar_shocks
Scripts to find self-similar solutions to converging and diverging shocks (see article https://doi.org/10.1063/5.0047518).

Currently:
- lazarus.py gives a valid result for converging shocks, including a graphic solution of the reflected shock.
- solver.py works for most cases (excluding inifinite-time converging shocks), but does not treat the reflected shock. It uses the analytic formula as a guess for lambda.

In the future, solver.py should try to converge to the correct lambda.
