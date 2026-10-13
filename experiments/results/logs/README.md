# Pipeline logs

Console logs of the study pipelines in [`experiments/scripts/`](../../scripts/README.md), copied from the GPU instances: [`t4/`](t4/) for the Tesla T4 runs, [`a10g/`](a10g/) for the A10G runs (plus the setup logs `t4/install.log` and `a10g/setup.log`). In the pipeline logs each stage starts with a `=== stage (time) ===` line.

- **Used by:** the run times in the [scripts README](../../scripts/README.md) and in deck appendix A6; the results READMEs link the log of each run.
- Raw aiperf artifacts stayed on the instances and are not here.
