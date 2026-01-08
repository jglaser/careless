name = "Careleast (Stochastic Phasing)"
description = "Options for the stochastic Langevin dynamics phasing engine."

args_and_kwargs = (
    (("--algorithm",), {
        "help": "Inference algorithm to use. 'variational' uses standard VI with surrogate posteriors. "
                "'careleast' uses stochastic Langevin dynamics on a real-space grid.",
        "type": str,
        "default": "variational",
        "choices": ["variational", "careleast"]
    }),
    (("--n-particles",), {
        "help": "Number of parallel chains (replicas) for the stochastic sampler. "
                "More particles allow for better exploration of multimodal solutions (e.g. enantiomorphs).",
        "type": int,
        "default": 4,
    }),
    (("--b-factor-prior",), {
        "help": "The Wilson B-factor used for the spectral preconditioner (smoothing kernel) in real space.",
        "type": float,
        "default": 20.0,
    }),
    (("--temperatures",), {
        "help": "Comma-separated list of temperatures for Parallel Tempering. "
                "Must match the number of particles. If not provided, a geometric progression is used.",
        "type": str,
        "default": None,
    }),
    (("--grid-oversampling",), {
        "help": "Oversampling factor for the real-space FFT grid relative to the Nyquist rate. "
                "Default is 1.5 (standard for density modification).",
        "type": float,
        "default": 1.5,
    }),
)
