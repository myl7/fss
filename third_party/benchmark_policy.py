PROFILES = ("decision", "smoke", "publication")
WORKLOADS = ("daily", "legacy")
RUNNERS = ("bench", "figure")
FULL_LAUNCHES_MIN = 1
FULL_LAUNCHES_MAX = 1024
DAILY_FULL_LAUNCHES = 1
LEGACY_FULL_LAUNCHES = 1024
FULL_EVAL_BATCH_KEYS = 1024
FULL_LAUNCHES_ENV = "FSS_BENCH_FULL_LAUNCHES"

TIMING = {
    "bench": {"smoke": (0.01, 1, 0.0), "decision": (0.3, 1, 0.0),
              "publication": (1.0, 5, 0.0)},
    "figure": {"smoke": (0.01, 1, 0.0), "decision": (0.3, 1, 0.0),
               "publication": (0.1, 5, 0.1)},
}

SELECTION = {
    "bench": {"decision": {"libraries": "main"}, "smoke": {"libraries": "main"},
              "publication": {"libraries": "all"}},
    "figure": {"decision": {"libraries": "fss", "domain_bits": (8, 20)},
               "smoke": {"libraries": "fss", "domain_bits": (8, 20)},
               "publication": {}},
}


def resolve_policy(args, runner):
    if runner not in RUNNERS:
        raise ValueError(f"unknown benchmark runner: {runner}")
    smoke_alias = bool(getattr(args, "smoke", False))
    explicit_profile = getattr(args, "profile", None)
    if smoke_alias:
        if explicit_profile and explicit_profile != "smoke":
            raise ValueError("--smoke is an alias for --profile smoke; the two profiles conflict")
        explicit_profile = "smoke"
    profile = explicit_profile or "decision"
    if profile not in PROFILES:
        raise ValueError(f"unknown benchmark profile: {profile}")
    workload = getattr(args, "workload", None) or "daily"
    if workload not in WORKLOADS:
        raise ValueError(f"unknown benchmark workload: {workload}")
    full_launches, launches_source = _resolve_full_launches(args, workload)
    resolved = {"profile": profile, "workload": workload,
                "full_launches": full_launches, "full_launches_env": FULL_LAUNCHES_ENV,
                "sources": {"profile": "smoke alias" if smoke_alias and profile == "smoke"
                            else "explicit" if explicit_profile else "default",
                            "workload": "explicit" if getattr(args, "workload", None) else "default",
                            "full_launches": launches_source}}
    minimum, repetitions, warmup = TIMING[runner][profile]
    for key, explicit, default in (("min_time_s", getattr(args, "min_time", None), minimum),
                                   ("repetitions", getattr(args, "repetitions", None), repetitions),
                                   ("warmup_s", getattr(args, "warmup", None), warmup)):
        resolved[key] = explicit if explicit is not None else default
        resolved["sources"][key] = "explicit" if explicit is not None else "profile"
    return resolved


INTERLEAVING_FLAG = "--benchmark_enable_random_interleaving"
INTERLEAVED_LIBRARIES = ("main", "fss")


def interleaving_enabled(library, repetitions, explicit=None):
    if explicit is not None:
        return explicit
    return library in INTERLEAVED_LIBRARIES and repetitions > 1


def interleaving_flag(library, repetitions, explicit=None):
    if library not in INTERLEAVED_LIBRARIES or (explicit is None and repetitions <= 1):
        return None
    return f"{INTERLEAVING_FLAG}={'true' if interleaving_enabled(library, repetitions, explicit) else 'false'}"


def _resolve_full_launches(args, workload):
    explicit = getattr(args, "full_launches", None)
    if explicit is not None and (not isinstance(explicit, int) or isinstance(explicit, bool)
                                 or not FULL_LAUNCHES_MIN <= explicit <= FULL_LAUNCHES_MAX):
        raise ValueError(f"--full-launches must be an integer from {FULL_LAUNCHES_MIN} "
                         f"to {FULL_LAUNCHES_MAX}")
    if workload == "legacy":
        if explicit is not None and explicit != LEGACY_FULL_LAUNCHES:
            raise ValueError(f"--workload legacy restores the historical {LEGACY_FULL_LAUNCHES} "
                             f"full launches; drop --full-launches or pass {LEGACY_FULL_LAUNCHES}")
        return LEGACY_FULL_LAUNCHES, "legacy workload"
    if explicit is not None:
        return explicit, "explicit"
    return DAILY_FULL_LAUNCHES, "default"


def apply_default_selection(args, runner, profile, policy):
    for field, value in SELECTION[runner].get(profile, {}).items():
        if getattr(args, field, None) is None:
            setattr(args, field, list(value) if isinstance(value, tuple) else value)
            policy["sources"][field] = "profile default"
