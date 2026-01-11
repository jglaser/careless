name = "Careleast (Stochastic Phasing)"
description = "Options for the stochastic Langevin dynamics phasing engine."

args_and_kwargs = (
    (("--algorithm",), {
        "help": "Inference algorithm to use.\n"
                "'variational': Standard VI with surrogate posteriors.\n"
                "'careleast_real': Stochastic SGLD on Real-Space Grid.\n"
                "'careleast_spectral': Stochastic SGLD on Full Fourier Grid.",
        "type": str,
        "default": "variational",
        "choices": ["variational", "careleast_real", "careleast_spectral"]
    }),
    (("--n-particles",), {
        "help": "Number of parallel chains (replicas). Default: 4",
        "type": int,
        "default": 4,
    }),
    (("--stochastic-points",), {
        "help": "Number of random real-space points to sample per step for constraints (Positivity/Sparsity) in sparse mode. "
                "Higher values improve constraint quality but slow down training. "
                "Set to 0 to disable. Default: 4096",
        "type": int,
        "default": 4096,
    }),
    (("--sparsity-weight",), {
        "help": "Weight for L1 sparsity prior (|rho|) on stochastic points. Promotes atomicity/phase extension. "
                "Default: 0.0",
        "type": float,
        "default": 0.0,
    }),
    (("--b-factor-prior",), {
        "help": "The Wilson B-factor for spectral preconditioning. Default: 20.0",
        "type": float,
        "default": 20.0,
    }),
    (("--temperatures",), {
        "help": "Comma-separated list of temperatures.",
        "type": str,
        "default": None,
    }),
    (("--grid-oversampling",), {
        "help": "Oversampling factor for FFT grid. Default: 1.5",
        "type": float,
        "default": 1.5,
    }),
    (("--disable-positivity",), {
        "help": "Disable real-space positivity constraint.",
        "action": "store_false",
        "dest": "use_positivity",
        "default": True,
    }),
    (("--tv-weight",), {
        "help": "Weight for Total Variation (TV) prior. Default: 0.0",
        "type": float,
        "default": 1e-4,
    }),
    (("--prior-weight",), {
        "help": "Scaling factor for prior energy. Default: 0.1",
        "type": float,
        "default": 0.1,
    }),
    (("--no-hermitian-symmetry",), {
        "help": "Disable explicit enforcement of Friedel symmetry in spectral mode.",
        "action": "store_false",
        "dest": "enforce_symmetry",
        "default": True,
    }),
)
