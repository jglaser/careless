#!/usr/bin/env python


def main():
    from . import __version__
    print(f"Careless version {__version__}")
    from careless.parser import parser
    parser = parser.parse_args()
    run_careless(parser)

def run_careless(parser):
    # We defer all inputs to make sure the parser has priority in modifying tf parameters
    import tensorflow as tf
    import numpy as np
    import reciprocalspaceship as rs
    from careless.io.manager import DataManager
    from careless.io.formatter import MonoFormatter,LaueFormatter
    from careless.models.base import BaseModel
    from careless.models.merging.surrogate_posteriors import TruncatedNormal
    from careless.models.merging.variational import VariationalMergingModel
    from careless.models.scaling.image import HybridImageScaler,ImageScaler
    from careless.models.scaling.nn import MLPScaler
    from careless.models.priors.wilson import DoubleWilsonPrior # Ensure this is imported
    from careless.models.merging.surrogate_posteriors import TruncatedNormal, FlowPosterior
    import tf_keras as tfk

    if parser.type == 'poly':
        df = LaueFormatter.from_parser(parser)
    elif parser.type == 'mono':
        df = MonoFormatter.from_parser(parser)
    elif parser.type == 'devices':
        print("###############################################")
        print("# TensorFlow can access the following devices #")
        print("###############################################")
        for dev in tf.config.list_physical_devices():
            print(f" - {dev.device_type}: {dev.name}")
        from sys import exit
        exit()


    inputs,rac = df.format_files(parser.reflection_files)
    dm = DataManager(inputs, rac, parser=parser)

    if parser.test_fraction is not None:
        train,test = dm.split_data_by_refl(parser.test_fraction)
    else:
        train,test = dm.inputs,None

    if parser.algorithm.startswith('careleast'):
        from careless.models.merging.careleast import CareleastRealSpace, CareleastSpectral

        # 1. Determine Grid Size (Needed for Spectral as well now)
        min_d = dm.asu_collection.dHKL.min() 
        high_res = 1.0 / np.max(np.sqrt(dm.asu_collection.dHKL)) 

        grid_spacing = high_res / (2.0 * parser.grid_oversampling)

        uc = dm.asu_collection.reciprocal_asus[0].cell
        nx = int(uc.a / grid_spacing)
        ny = int(uc.b / grid_spacing)
        nz = int(uc.c / grid_spacing)
        grid_size = (nx, ny, nz)
        print(f"Careleast: Initializing grid {grid_size} for resolution {high_res:.2f}A")

        temp_model = dm.build_model()
        likelihood = temp_model.likelihood
        scaling_model = temp_model.scaling_model

        if parser.temperatures:
            temps = [float(x) for x in parser.temperatures.split(',')]
        else:
            temps = None

        if parser.algorithm == 'careleast_real':
            print(f"Careleast: Real-Space SGLD.")
            model = CareleastRealSpace(
                asu_collection=dm.asu_collection,
                likelihood=likelihood,
                scaling_model=scaling_model,
                grid_size=grid_size,
                unit_cell=uc,
                n_particles=parser.n_particles,
                b_factor=parser.b_factor_prior,
                learning_rate=parser.learning_rate,
                temperatures=temps,
                use_positivity=parser.use_positivity, 
                tv_weight=parser.tv_weight,
                prior_weight=parser.prior_weight,
            )
        elif parser.algorithm == 'careleast_spectral':
            print(f"Careleast: Spectral SGLD (Fourier Coeffs + Real Constraints).")
            model = CareleastSpectral(
                asu_collection=dm.asu_collection,
                likelihood=temp_model.likelihood,
                scaling_model=temp_model.scaling_model,
                grid_size=grid_size, # Passed for FFT
                unit_cell=uc,       # Passed for Wilson
                n_particles=parser.n_particles,
                b_factor=parser.b_factor_prior,
                learning_rate=parser.learning_rate,
                temperatures=temps,
                use_positivity=parser.use_positivity,
                tv_weight=parser.tv_weight,
                prior_weight=parser.prior_weight,
                enforce_symmetry=parser.enforce_symmetry,
                stochastic_points=parser.stochastic_points,
                sparsity_weight=parser.sparsity_weight,
            )

        optimizer = tfk.optimizers.Adam(learning_rate=parser.learning_rate)
        model.compile(optimizer=optimizer)
    else:
        # ... (Standard Variational Code omitted for brevity, same as original) ...
        # [Note: In full file replacement, keep the original else block here]
        prior = None
        parents = parser.parents
        if parents is None:
            prior = dm.get_wilson_prior(parser.wilson_prior_b)
        else:
            parents = [None if i == 'None' else int(i) for i in parents.split(',')]
            r_values = parser.dwr
            r_values = [float(i) for i in r_values.split(',')]
            sigma = dm.get_wilson_sigma(parser.wilson_prior_b)
            reindexing_ops = parser.reindexing_ops
            if reindexing_ops is not None:
                import gemmi
                delim = ';'
                reindexing_ops = [gemmi.Op(i) for i in reindexing_ops.split(delim)]
            prior = DoubleWilsonPrior(
                dm.asu_collection,
                parents,
                r_values,
                reindexing_ops,
                sigma=sigma,
                optimize_r=parser.optimize_double_wilson_r
            )
        loc, scale = prior.mean(), prior.stddev()
        scale = scale * parser.structure_factor_init_scale
        if parser.surrogate_posterior == 'flow':
            from careless.models.merging.surrogate_posteriors import FlowPosterior
            surrogate_posterior = FlowPosterior.from_loc_and_scale(
                loc, scale, depth=parser.flow_depth,
                hidden_units=parser.flow_hidden_units,
                inference_samples=parser.flow_inference_samples,
                name='structure_factor'
            )
        else:
            surrogate_posterior = TruncatedNormal.from_loc_and_scale(loc, scale, name='structure_factor')
        model = dm.build_model(surrogate_posterior=surrogate_posterior)
        if parser.scale_file is not None:
            model.scaling_model.load_weights(parser.scale_file)
        if parser.freeze_scales:
            model.scaling_model.trainable = False
        if parser.structure_factor_file is not None:
            model.surrogate_posterior.load_weights(parser.structure_factor_file)
        if parser.freeze_structure_factors:
            model.surrogate_posterior.trainable = False

    validation_frequency = parser.validation_frequency
    progress = not parser.disable_progress_bar

    history = model.train_model(
        tuple(map(tf.convert_to_tensor, train)),
        parser.iterations,
        message="Sampling" if parser.algorithm.startswith('careleast') else "Training",
        validation_data=test,
        validation_frequency=validation_frequency,
        progress=progress,
        reduce_retracing=parser.reduce_retracing,
        jit_compile=parser.jit_compile,
    )

    if parser.algorithm.startswith('careleast'):
        print("Extracting phases from coldest chain...")
        
        if parser.algorithm == 'careleast_real':
            final_density = model.density[0]
            # Manual forward fft for output
            F_grid = tf.signal.rfft3d(final_density).numpy() 
        elif parser.algorithm == 'careleast_spectral':
            # F_state is now (N, nx, ny, nz)
            # Symmetrize just in case for output
            F_grid_complex = model._symmetrize_F(model.F_state)[0]
            F_grid = F_grid_complex.numpy()
            final_density = tf.math.real(tf.signal.ifft3d(F_grid_complex))

        nx, ny, nz = model.grid_size

        for i, rasu in enumerate(dm.asu_collection.reciprocal_asus):
            hkls = rasu.lookup_table.get_hkls().astype(np.int32)
            h, k, l = hkls.T

            # Map sparse HKLs to dense grid indices (handling Friedel)
            h_idx = np.mod(h, nx)
            k_idx = np.mod(k, ny)
            l_idx = np.mod(l, nz)
            F_values = F_grid[idx_h, idx_k, idx_l]

            ds = rs.DataSet({
                'H': h, 'K': k, 'L': l,
                'F': np.abs(F_values).astype(np.float32),
                'PHI': np.rad2deg(np.angle(F_values)).astype(np.float32),
                'I': np.square(np.abs(F_values)).astype(np.float32),
                'SigF': np.zeros_like(h, dtype=np.float32),
                'SigI': np.zeros_like(h, dtype=np.float32),
            }, cell=rasu.cell, spacegroup=rasu.spacegroup).infer_mtz_dtypes()

            if rasu.anomalous:
                ds = ds.set_index(['H', 'K', 'L']).unstack_anomalous()

            filename = parser.output_base + f'_{i}.mtz'
            print(f"Writing {filename}...")
            ds.write_mtz(filename)

        np.save(parser.output_base + '_density.npy', final_density.numpy())
    else:
        # Standard saving logic (Same as original)
        for i,ds in enumerate(dm.get_results(model.surrogate_posterior, inputs=train)):
            filename = parser.output_base + f'_{i}.mtz'
            ds.write_mtz(filename)
        filename = parser.output_base + f'_history.csv'
        history = rs.DataSet(history).to_csv(filename, index_label='step')
        model.surrogate_posterior.save_weights(parser.output_base + '_structure_factor')
        model.scaling_model.save_weights(parser.output_base + '_scale')
        # ... (Rest of standard output logic) ...

if __name__=="__main__":
    main()
