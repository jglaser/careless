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
                "Default: 4",
        "type": int,
        "default": 4,
    }),
    (("--b-factor-prior",), {
        "help": "The Wilson B-factor used for the spectral preconditioner (smoothing kernel) in real space. "
                "Default: 20.0",
        "type": float,
        "default": 20.0,
    }),
    (("--temperatures",), {
        "help": "Comma-separated list of temperatures for Parallel Tempering. "
                "If not provided, a geometric progression is used.",
        "type": str,
        "default": None,
    }),
    (("--grid-oversampling",), {
        "help": "Oversampling factor for the real-space FFT grid relative to the Nyquist rate. "
                "Default: 1.5",
        "type": float,
        "default": 1.5,
    }),
    (("--disable-positivity",), {
        "help": "Disable the real-space positivity constraint. Use this for Neutron diffraction data.",
        "action": "store_false",
        "dest": "use_positivity",
        "default": True,
    }),
)
