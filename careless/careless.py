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

    if parser.algorithm == 'careleast':
        from careless.models.merging.careleast import CareleastRealSpace

        # 1. Determine Grid Size from Resolution
        min_d = dm.asu_collection.dHKL.min() # d_spacing is inverse? Check dHKL definition.
        # usually dHKL in careless is 1/d^2 or just d.
        # Let's assume we can get high res limit.
        high_res = 1.0 / np.max(np.sqrt(dm.asu_collection.dHKL)) # Assuming dHKL is 1/d^2

        # Grid sampling: Resolution / 2 (Nyquist) / Oversampling
        grid_spacing = high_res / (2.0 * parser.grid_oversampling)

        uc = dm.asu_collection.reciprocal_asus[0].cell
        nx = int(uc.a / grid_spacing)
        ny = int(uc.b / grid_spacing)
        nz = int(uc.c / grid_spacing)
        grid_size = (nx, ny, nz)
        print(f"Careleast: Initializing real-space grid {grid_size} for resolution {high_res:.2f}A")

        # 2. Build Likelihood & Scaling (Reuse DataManager logic manually or refactor)
        # We can reuse dm.build_model() to get the components, then discard the VAE
        temp_model = dm.build_model()
        likelihood = temp_model.likelihood
        scaling_model = temp_model.scaling_model

        # 3. Parse Temperatures
        if parser.temperatures:
            temps = [float(x) for x in parser.temperatures.split(',')]
        else:
            temps = None

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
            use_positivity=parser.use_positivity, # Controlled by --disable-positivity
        )

        # Compile needed for fit()
        optimizer = tfk.optimizers.SGD(
            learning_rate=parser.learning_rate,
            global_clipnorm=1.0 # Clips gradients to norm 1.0
        )
        model.compile(optimizer=optimizer)
    else:
        prior = None
        parents = parser.parents
        if parents is None:
            prior = dm.get_wilson_prior(parser.wilson_prior_b)
        else:
            # Double Wilson Prior Logic
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

        # Now loc and scale are available
        loc, scale = prior.mean(), prior.stddev()
        scale = scale * parser.structure_factor_init_scale

        if parser.surrogate_posterior == 'flow':
            # Use Flow Posterior
            # Note: scale_shift logic from TruncatedNormal handled internally or via base_scale init
            from careless.models.merging.surrogate_posteriors import FlowPosterior
            surrogate_posterior = FlowPosterior.from_loc_and_scale(
                loc,
                scale,
                depth=parser.flow_depth,
                hidden_units=parser.flow_hidden_units,
                inference_samples=parser.flow_inference_samples,
                name='structure_factor'
            )
            print(f"Initialized FlowPosterior with depth={parser.flow_depth}, hidden_units={parser.flow_hidden_units}")
        else:
            # Default Truncated Normal
            surrogate_posterior = TruncatedNormal.from_loc_and_scale(
                loc,
                scale,
                name='structure_factor'
            )

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
        message="Sampling" if parser.algorithm=='careleast' else "Training",
        validation_data=test,
        validation_frequency=validation_frequency,
        progress=progress,
        reduce_retracing=parser.reduce_retracing,
        jit_compile=parser.jit_compile,
    )

    if parser.algorithm == 'careleast':
        # --- CARELEAST RESULT EXTRACTION ---
        print("Extracting phases from coldest chain...")
        final_density = model.density[0] # Coldest chain (Index 0)

        # Forward FFT to getting Complex F on P1 grid
        F_grid = model._fft_forward(final_density[None, ...])[0]
        F_grid_np = F_grid.numpy()
        nx, ny, nz = model.grid_size

        for i, rasu in enumerate(dm.asu_collection.reciprocal_asus):
            # Get unique HKLs for this ASU
            hkls = rasu.lookup_table.get_hkls().astype(np.int32)
            h, k, l = hkls.T

            # Map sparse HKLs to dense grid indices (handling wrapping)
            h_idx = np.mod(h, nx)
            k_idx = np.mod(k, ny)
            l_idx = np.mod(l, nz)

            # Gather complex Structure Factors
            F_values = F_grid_np[h_idx, k_idx, l_idx]

            # Build Output DataSet
            ds = rs.DataSet({
                'H': h,
                'K': k,
                'L': l,
                'F': np.abs(F_values).astype(np.float32),
                'PHI': np.rad2deg(np.angle(F_values)).astype(np.float32),
                'I': np.square(np.abs(F_values)).astype(np.float32),
                # Sigmas are unknown/undefined for single-sample extraction
                'SigF': np.zeros_like(h, dtype=np.float32),
                'SigI': np.zeros_like(h, dtype=np.float32),
            }, cell=rasu.cell, spacegroup=rasu.spacegroup).infer_mtz_dtypes()

            # Reformat anomalous data if necessary
            if rasu.anomalous:
                ds = ds.set_index(['H', 'K', 'L']).unstack_anomalous()

            filename = parser.output_base + f'_{i}.mtz'
            print(f"Writing {filename}...")
            ds.write_mtz(filename)

        # Also save the raw density map (MRC/CCP4 format equivalent)
        # We can dump it to a numpy file or simple map file for inspection
        np.save(parser.output_base + '_density.npy', final_density.numpy())
    else:
        for i,ds in enumerate(dm.get_results(model.surrogate_posterior, inputs=train)):
            filename = parser.output_base + f'_{i}.mtz'
            ds.write_mtz(filename)

        filename = parser.output_base + f'_history.csv'
        history = rs.DataSet(history).to_csv(filename, index_label='step')

        model.surrogate_posterior.save_weights(parser.output_base + '_structure_factor')
        model.scaling_model.save_weights(parser.output_base + '_scale')
        if parser.save_data_manager:
            import pickle
            with open(parser.output_base + "_data_manager.pickle", "wb") as out:
                pickle.dump(dm, out)

        predictions_data = None
        if test is not None:
            for file_id, (ds_train, ds_test) in enumerate(zip(
                    dm.get_predictions(model, train, test_value=0),
                    dm.get_predictions(model, test, test_value=1),
                    )):
                filename = parser.output_base + f'_predictions_{file_id}.mtz'
                rs.concat((
                    ds_train,
                    ds_test,
                )).write_mtz(filename)
        else:
            for file_id, ds_train in enumerate(dm.get_predictions(model, train, test_value=0)):
                filename = parser.output_base + f'_predictions_{file_id}.mtz'
                ds_train.write_mtz(filename)

        if parser.merge_half_datasets:
            scaling_model = model.scaling_model
            scaling_model.trainable = False
            xval_data = [None] * len(dm.asu_collection)
            for repeat in range(parser.half_dataset_repeats):
                for half_id, half in enumerate(dm.split_data_by_image()):
                    model = dm.build_model(scaling_model=scaling_model)
                    history = model.train_model(
                        tuple(map(tf.convert_to_tensor, half)), 
                        parser.iterations,
                        message=f"Merging repeat {repeat+1} half {half_id+1}",
                        progress=progress,
                        reduce_retracing=parser.reduce_retracing,
                        jit_compile=parser.jit_compile,
                    )

                    for file_id,ds in enumerate(dm.get_results(model.surrogate_posterior, inputs=half)):
                        ds['repeat'] = rs.DataSeries(repeat, index=ds.index, dtype='I')
                        ds['half'] = rs.DataSeries(half_id, index=ds.index, dtype='I')
                        if xval_data[file_id] is None:
                            xval_data[file_id] = ds
                        else:
                            xval_data[file_id] = rs.concat((xval_data[file_id], ds))

            for file_id, ds in enumerate(xval_data):
                filename = parser.output_base + f'_xval_{file_id}.mtz'
                ds.write_mtz(filename)

    if parser.embed:
        from IPython import embed
        embed(colors='Linux')


if __name__=="__main__":
    main()

