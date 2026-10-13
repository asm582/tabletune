"""In-engine Dynamic SD controller: hold K near table edges, re-fit the table live.

No vLLM imports; vllm_plugin.py attaches it to the scheduler, which calls
choose_k(batch_size) every step, observe(drafted, accepted) per request after
verification, and observe_no_proposal() for decode steps without drafts.

- Hysteresis: keep the current K while the table still picks it within
  `hysteresis` batch sizes of the live batch, so K doesn't flip every step
  when the batch hovers at a table edge.
- Re-fit: every `refit_steps` steps, rebuild the table with core.make_schedule
  from the tuner's step costs and live acceptance blended with the tuner's
  prior, a_i = (w * prior_i + accepted_i) / (w + drafted_i), so positions that
  are rarely drafted keep the prior. Only K values in the starting table are
  used: those have CUDA graphs.

The other options (explore, prior decay, any K) are experimental and
off by default; README.md lists them. Changing K never changes the output
tokens: verification is lossless.

Failure policy: invalid settings raise ValueError when the controller is built,
so vLLM stops at startup. While serving nothing raises: a failed re-fit keeps
the current table, and MAX_REFIT_ERRORS failures in a row turn re-fit off.
"""

from __future__ import annotations

import logging
import os

from ..core import (PROPOSAL_RATE_FULL, Costs, expand, load_acceptance,
                   load_costs, make_schedule, step_ms, validate_acceptance,
                   validate_costs)

logger = logging.getLogger(__name__)

# Methods whose drafter proposes only on some steps.
SOMETIMES_PROPOSING = {"ngram"}
# Consecutive failed re-fits after which re-fitting is turned off.
MAX_REFIT_ERRORS = 3


def _check(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(f"Dynamic SD controller: {msg}")


def extend_prior(prior: list[float], n: int) -> list[float]:
    """Extend a prior to n positions by continuing its last ratio.

    Missing positions must stay positive: with a zero prior those K values are
    never chosen, so never drafted and never learned.
    """
    if not prior or len(prior) >= n:
        return prior
    r = prior[-1] / prior[-2] if len(prior) >= 2 and prior[-2] > 0 else 1.0
    out = list(prior)
    while len(out) < n:
        out.append(out[-1] * min(r, 1.0))
    return out


class DynamicSDController:

    def __init__(
        self,
        lookup: list[int],
        num_spec_tokens: int,
        *,
        hysteresis: int = 0,
        costs: Costs | None = None,
        prior_accept: list[float] | None = None,
        refit_steps: int = 0,
        prior_weight: float = 2000.0,
        decay: float = 0.5,
        log_steps: int = 0,
        track_proposals: bool = False,
        prior_proposal_rate: float = 1.0,
        explore_every: int = 0,
        explore_max_cost: float = 0.25,
        prior_decay: float = 0.0,
        any_k: bool = False,
    ) -> None:
        if len(lookup) < 2:
            raise ValueError("lookup must cover at least batch size 1 (index 0 is unused)")
        _check(isinstance(num_spec_tokens, int) and num_spec_tokens >= 0,
               f"num_spec_tokens must be an integer >= 0, got {num_spec_tokens!r}")
        _check(all(isinstance(k, int) for k in lookup[1:]), "lookup entries must be integers")
        for name, value in (("prior_weight", prior_weight), ("explore_max_cost", explore_max_cost)):
            _check(isinstance(value, (int, float)) and 0 <= value < float("inf"),
                   f"{name} must be a finite number >= 0, got {value!r}")
        _check(0 <= decay <= 1, f"decay must be in [0, 1], got {decay!r}")
        _check(0 <= prior_decay < 1, f"prior_decay must be in [0, 1), got {prior_decay!r}")
        _check(0 < prior_proposal_rate <= 1,
               f"prior_proposal_rate must be in (0, 1], got {prior_proposal_rate!r}")
        for name, value in (("refit_steps", refit_steps), ("explore_every", explore_every),
                            ("log_steps", log_steps)):
            _check(isinstance(value, int) and value >= 0,
                   f"{name} must be an integer >= 0, got {value!r}")
        if prior_accept:
            try:
                validate_acceptance(list(prior_accept))
            except ValueError as e:
                raise ValueError(f"Dynamic SD controller: prior: {e}") from None

        # Never more than vLLM drafts, never negative.
        self.lookup = [max(0, min(k, num_spec_tokens)) for k in lookup]
        self.max_b = len(lookup) - 1
        self.num_spec_tokens = num_spec_tokens
        self.allowed_ks = (list(range(num_spec_tokens + 1)) if any_k
                           else sorted(set(self.lookup[1:])))
        if costs:
            try:
                validate_costs(costs)
            except ValueError as e:
                raise ValueError(f"Dynamic SD controller: {e}") from None
            table_ks = set(self.lookup[1:])
            _check(table_ks <= set(costs), f"step costs cover K={sorted(costs)}, "
                   f"but the table uses K={sorted(table_ks)}")
            if any_k:  # a re-fit can only weigh K values it has costs for
                self.allowed_ks = [k for k in self.allowed_ks if k in costs]
        self.hysteresis = max(0, hysteresis)
        self.log_steps = log_steps

        self.refit_steps = refit_steps if costs and prior_accept else 0
        self.refit_errors = 0        # consecutive failed re-fits
        self.costs = costs
        self.prior = extend_prior(list(prior_accept or [])[:num_spec_tokens], num_spec_tokens)
        self.prior_weight = prior_weight
        self.decay = decay
        self.drafted = [0.0] * num_spec_tokens
        self.accepted = [0.0] * num_spec_tokens
        # Proposal rate (n-gram): vLLM reports only steps with drafts, so decode
        # steps without drafts are counted here too.
        self.track_proposals = track_proposals
        self.prior_proposal_rate = prior_proposal_rate
        self.decode_steps = 0.0      # request-steps with K > 0
        self.proposal_steps = 0.0    # ... of which had drafts

        # Experimental options (README.md).
        self.explore_every = explore_every if costs else 0
        self.explore_max_cost = explore_max_cost
        self.prior_decay = prior_decay
        self.prior_weight_floor = 0.01 * prior_weight

        self.current_k: int | None = None
        self.steps = 0
        self.stats = {"steps": 0, "switches": 0, "held": 0, "refits": 0, "table_changes": 0}
        if self.explore_every:
            self.stats["explored"] = 0

    # ------------------------------------------------------------------ K choice
    def choose_k(self, batch_size: int) -> int:
        """K for this step. Called once per scheduling step."""
        self.steps += 1
        self.stats["steps"] += 1
        if self.refit_steps and self.steps % self.refit_steps == 0:
            self._refit_safely()
        if self.log_steps and self.steps % self.log_steps == 0:
            logger.info("Dynamic SD controller: %s", dict(self.stats))   # a snapshot, not the live dict

        b = min(max(batch_size, 1), self.max_b)
        target = self.lookup[b]
        if self.current_k is None or target == self.current_k:
            self.current_k = target
        elif self._still_nearby(b, self.current_k):
            self.stats["held"] += 1
        else:
            self.current_k = target
            self.stats["switches"] += 1
        k = self.current_k
        if self.explore_every and self.steps % self.explore_every == 0:
            k = self._explore(b, k)
        return k

    def _refit_safely(self) -> None:
        """refit() for the serving path: an error keeps the current table."""
        try:
            self.refit()
            self.refit_errors = 0
        except Exception as e:  # noqa: BLE001 - the scheduler must keep running
            self.refit_errors += 1
            self.stats["refit_errors"] = self.stats.get("refit_errors", 0) + 1
            off = self.refit_errors >= MAX_REFIT_ERRORS
            if off:
                self.refit_steps = 0
            logger.warning("Dynamic SD controller: re-fit failed (%s: %s); %s", type(e).__name__, e,
                           f"re-fit turned off after {self.refit_errors} failures in a row, "
                           "keeping the current table" if off else "keeping the current table")

    def describe(self) -> str:
        """One line with the active settings, logged by the plugin at startup."""
        return (f"Dynamic SD controller enabled: hysteresis={self.hysteresis}, "
                f"refit={'every %d steps' % self.refit_steps if self.refit_steps else 'off'}, "
                f"allowed K={self.allowed_ks}"
                + (", tracking proposal rate" if self.track_proposals else "")
                + (f", explore every {self.explore_every} steps (max +{self.explore_max_cost:.0%})"
                   if self.explore_every else "")
                + (f", prior decay {self.prior_decay}" if self.prior_decay else ""))

    def _explore(self, b: int, k: int) -> int:
        """The deepest allowed K for this one step, if it costs at most
        explore_max_cost more. Gives positions the table never drafts some live
        data, so a pessimistic prior cannot lock deep K out."""
        deep = self.allowed_ks[-1]
        if deep <= k or deep not in self.costs or k not in self.costs:
            return k
        if step_ms(self.costs, deep, b) > (1 + self.explore_max_cost) * step_ms(self.costs, k, b):
            return k
        self.stats["explored"] += 1
        return deep

    def _still_nearby(self, b: int, k: int) -> bool:
        """True if the table picks k for some batch size within `hysteresis` of b."""
        lo = max(1, b - self.hysteresis)
        hi = min(self.max_b, b + self.hysteresis)
        return any(self.lookup[x] == k for x in range(lo, hi + 1))

    # ------------------------------------------------------------ acceptance
    def observe(self, num_drafted: int, num_accepted: int) -> None:
        """One request's verification result for one step. Impossible counts
        (negative, or more accepted than drafted) are clamped."""
        num_drafted = max(0, int(num_drafted))
        num_accepted = max(0, min(int(num_accepted), num_drafted))
        for i in range(min(num_drafted, self.num_spec_tokens)):
            self.drafted[i] += 1
        for i in range(min(num_accepted, num_drafted, self.num_spec_tokens)):
            self.accepted[i] += 1
        if self.track_proposals and num_drafted > 0:
            self.decode_steps += 1
            self.proposal_steps += 1

    def observe_no_proposal(self) -> None:
        """A request's decode step without drafts. Counts only for drafters that
        propose on some steps, and only while K > 0 (at K=0 none are expected)."""
        if self.track_proposals and (self.current_k or 0) > 0:
            self.decode_steps += 1

    def proposal_rate_estimate(self) -> float:
        """Share of decode steps with drafts, blended with the prior like acceptance."""
        if not self.track_proposals:
            return 1.0
        w = self.prior_weight
        if w + self.decode_steps <= 0:           # prior_weight=0 and no data yet
            return self.prior_proposal_rate
        p = (w * self.prior_proposal_rate + self.proposal_steps) / (w + self.decode_steps)
        # make_schedule needs p > 0; a drafter that has never proposed gets a
        # tiny rate, which already makes the cheapest K the best.
        p = max(p, 1e-3)
        return 1.0 if p >= PROPOSAL_RATE_FULL else p

    def acceptance_estimate(self) -> list[float]:
        """Per-position acceptance: live counts blended with the prior."""
        w = self.prior_weight
        return [(w * p + a) / (w + d) if w + d > 0 else p
                for p, a, d in zip(self.prior, self.accepted, self.drafted)]

    # ----------------------------------------------------------------- re-fit
    def refit(self) -> list[list[int]] | None:
        """Rebuild the table from stored costs and live acceptance."""
        if not (self.costs and self.prior):
            return None
        est = self.acceptance_estimate()
        rate = self.proposal_rate_estimate()
        schedule = make_schedule(self.costs, est, self.max_b, allowed_ks=self.allowed_ks,
                                 proposal_rate=rate)
        new_lookup = [min(k, self.num_spec_tokens) for k in expand(schedule, self.max_b)]
        stray = set(new_lookup[1:]) - set(self.allowed_ks)
        if stray:  # never switch to a K without a captured CUDA graph
            raise RuntimeError(f"re-fit chose K={sorted(stray)}, allowed K={self.allowed_ks}")
        self.stats["refits"] += 1
        if new_lookup != self.lookup:
            self.lookup = new_lookup
            self.stats["table_changes"] += 1
            logger.info("Dynamic SD controller: new table %s (acceptance by position %s%s)", schedule,
                        [round(x, 3) for x in est],
                        f", proposal rate {rate:.3f}" if self.track_proposals else "")
        # Forget old evidence gradually so the estimate follows workload drift.
        if self.prior_decay:
            self.prior_weight = max(self.prior_weight * (1 - self.prior_decay),
                                    self.prior_weight_floor)
        keep = 1 - self.decay
        self.drafted = [d * keep for d in self.drafted]
        self.accepted = [a * keep for a in self.accepted]
        self.decode_steps *= 1 - self.decay
        self.proposal_steps *= 1 - self.decay
        return schedule


_TRUE, _FALSE = ("1", "true", "yes", "on"), ("0", "false", "no", "off")


def _flag(env: dict[str, str], name: str, default: bool) -> bool:
    """A 0/1 switch; unset or empty gives `default`."""
    raw = env.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in _TRUE or raw in _FALSE:
        return raw in _TRUE
    raise ValueError(f"Dynamic SD controller: {name}={env[name]!r}: expected 0 or 1")


def _number(env: dict[str, str], name: str, default: int | float, kind: type) -> int | float:
    """An int or float setting; the range is checked by DynamicSDController."""
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return kind(raw)
    except ValueError:
        raise ValueError(f"Dynamic SD controller: {name}={raw!r}: expected "
                         f"{'an integer' if kind is int else 'a number'}") from None


def controller_from_env(lookup: list[int], num_spec_tokens: int,
                        env: dict[str, str] | None = None,
                        method: str | None = None) -> DynamicSDController | None:
    """Build a controller from the VLLM_DSD_* variables (listed in README.md).

    Returns None unless VLLM_DSD_CONTROLLER is set. Once enabled, a bad value,
    a missing file, or only one of VLLM_DSD_COSTS / VLLM_DSD_PRIOR raises
    ValueError naming the variable.
    """
    env = os.environ if env is None else env
    if not _flag(env, "VLLM_DSD_CONTROLLER", False):
        return None
    costs, prior, prior_rate = None, None, 1.0
    has_costs, has_prior = bool(env.get("VLLM_DSD_COSTS")), bool(env.get("VLLM_DSD_PRIOR"))
    if has_costs != has_prior:
        given, missing = (("VLLM_DSD_COSTS", "VLLM_DSD_PRIOR") if has_costs
                          else ("VLLM_DSD_PRIOR", "VLLM_DSD_COSTS"))
        raise ValueError(f"Dynamic SD controller: {given} is set but {missing} is not; "
                         "re-fit needs both")
    if has_costs:
        try:
            costs = load_costs(env["VLLM_DSD_COSTS"])
        except (OSError, ValueError) as e:
            raise ValueError(f"Dynamic SD controller: VLLM_DSD_COSTS: {e}") from None
        try:
            prior, prior_rate = load_acceptance(env["VLLM_DSD_PRIOR"])
        except (OSError, ValueError) as e:
            raise ValueError(f"Dynamic SD controller: VLLM_DSD_PRIOR: {e}") from None
    any_k = _flag(env, "VLLM_DSD_ANY_K", False)
    if any_k and env.get("VLLM_USE_V2_MODEL_RUNNER", "").lower() not in ("0", "false"):
        # MRv2 records FULL CUDA graphs per K: only the table's K values have them.
        any_k = False
        logger.warning("Dynamic SD controller: VLLM_DSD_ANY_K ignored (needs VLLM_USE_V2_MODEL_RUNNER=0)")
    track_proposals = method in SOMETIMES_PROPOSING
    return DynamicSDController(
        lookup, num_spec_tokens,
        hysteresis=_number(env, "VLLM_DSD_HYSTERESIS", 4, int),
        costs=costs, prior_accept=prior,
        refit_steps=_number(env, "VLLM_DSD_REFIT_STEPS", 500, int),
        prior_weight=_number(env, "VLLM_DSD_PRIOR_WEIGHT", 2000.0, float),
        decay=_number(env, "VLLM_DSD_DECAY", 0.5, float),
        log_steps=_number(env, "VLLM_DSD_LOG_STEPS", 0, int),
        track_proposals=track_proposals, prior_proposal_rate=min(prior_rate, 1.0),
        explore_every=_number(env, "VLLM_DSD_EXPLORE", 0, int),
        explore_max_cost=_number(env, "VLLM_DSD_EXPLORE_MAX_COST", 0.25, float),
        prior_decay=_number(env, "VLLM_DSD_PRIOR_DECAY", 0.0, float),
        any_k=any_k,
    )
