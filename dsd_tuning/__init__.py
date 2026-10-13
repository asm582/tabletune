"""dsd_tuning: measure step time, measure acceptance, apply one formula.

tuner/ builds the table offline (the profilers need a GPU and vLLM); controller/
keeps it right inside vLLM; core.py holds the formula both use (no GPU needed).
See dsd_tuning/README.md.
"""
