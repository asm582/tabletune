"""Repo-relative locations shared by the H100 analysis and figure scripts."""
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
RESULTS = os.path.join(REPO, "experiments", "results")
SWEEPS = os.path.join(RESULTS, "sweeps")                 # every GPU's sweeps: T4, A10G, H100 V1 (-h100), H100 V2 (-h100v2)
STUDIES = os.path.join(RESULTS, "h100-studies")          # baselines, MoE, credibility, llm-d, workload switch / jumping load
OUT = os.environ.get("FIG_OUT", os.path.join(REPO, "experiments", "results", "figures"))  # figures; FIG_OUT overrides
