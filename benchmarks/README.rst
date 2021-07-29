Airspeed Velocity Benchmarks
====================================


This directory contains benchmarks written in the structure supported by airspeed velocity.

We assume that you have Airspeed Velocity installed. If not, try::

    pip install asv


To run the benchmarks (against the master), run::

    asv run

If you are running it for the first time on a machine, it will ask you a set of questions to capture the details of the machine.
We assume that that you have ``conda`` available. A ``conda`` environment is constructed under ``.asv/env`` directory to run the benchmarks.
The benchmarking results wlil be stored inside ``.asv/results`` directory. 

To see the results for the master, run::

    asv show master
