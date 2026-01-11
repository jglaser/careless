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
            
            initial_F_grid = None
            if hasattr(parser, 'initial_mtz') and parser.initial_mtz:
                print(f"Loading initialization from {parser.initial_mtz}...")
                import reciprocalspaceship as rs
                
                # --- FIX: Add .reset_index() here ---
                ds_init = rs.read_mtz(parser.initial_mtz).reset_index()
                
                # Now these accessors will work correctly
                h, k, l = ds_init.H.to_numpy(), ds_init.K.to_numpy(), ds_init.L.to_numpy()
                
                # (The rest of the logic remains the same)
                if 'FC' in ds_init.columns:
                    amp = ds_init['FC'].to_numpy()
                    phi = np.deg2rad(ds_init['PHIFC'].to_numpy())
                else: 
                    amp = ds_init['F'].to_numpy()
                    phi = np.deg2rad(ds_init['PHI'].to_numpy())
                    
                F_complex = amp * np.exp(1j * phi)
                
                # Create Blank Grid (nx, ny, nz_half)
                initial_F_grid = np.random.normal(0, 1e-6, (nx, ny, nz // 2 + 1)).astype(np.complex64)
                initial_F_grid = initial_F_grid + 1j * np.random.normal(0, 1e-6, (nx, ny, nz // 2 + 1))
                
                # 1. Handle Positive L
                mask_pos = l >= 0
                h_pos, k_pos, l_pos = h[mask_pos], k[mask_pos], l[mask_pos]
                F_pos = F_complex[mask_pos]
                
                idx_h = np.mod(h_pos, nx)
                idx_k = np.mod(k_pos, ny)
                idx_l = l_pos 
                
                valid = idx_l < (nz // 2 + 1)
                initial_F_grid[idx_h[valid], idx_k[valid], idx_l[valid]] = F_pos[valid]
                
                # 2. Handle Negative L (Friedel Mates)
                mask_neg = l < 0
                h_neg, k_neg, l_neg = h[mask_neg], k[mask_neg], l[mask_neg]
                F_neg = F_complex[mask_neg]
                
                idx_h_f = np.mod(-h_neg, nx)
                idx_k_f = np.mod(-k_neg, ny)
                idx_l_f = -l_neg
                
                F_neg_conj = np.conj(F_neg)
                
                valid_f = idx_l_f < (nz // 2 + 1)
                initial_F_grid[idx_h_f[valid_f], idx_k_f[valid_f], idx_l_f[valid_f]] = F_neg_conj[valid_f]

            model = CareleastSpectral(
                asu_collection=dm.asu_collection,
                likelihood=temp_model.likelihood,
                scaling_model=temp_model.scaling_model,
                grid_size=grid_size,
                unit_cell=uc,
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
                initial_F=initial_F_grid, # Pass the grid
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
        import gemmi

        if parser.algorithm == 'careleast_real':
            # Real Space Model (Density is the state)
            final_density = model.density[0]
            # FFT to get Structure Factors for MTZ
            F_grid = tf.signal.rfft3d(final_density).numpy()

        elif parser.algorithm == 'careleast_spectral':
            # Spectral Model (State is flattened Half-Grid F)
            # 1. Reshape flat state to (N_particles, nx, ny, nz_half)
            # Note: model.F_state corresponds to the output of rfft3d
            F_half_grid = tf.reshape(model.F_state, (model.n_particles, *model.grid_size[:-1], model.nz_half))

            # 2. Use irfft3d to reconstruct real-space density correctly
            # This function expects the half-grid format (nx, ny, nz/2 + 1)
            # and handles the reconstruction of negative frequencies internally.
            final_density = tf.signal.irfft3d(F_half_grid[0])

            # 3. For the MTZ, we still need the complex F values on the half-grid
            F_grid = F_half_grid[0].numpy()

        nx, ny, nz = model.grid_size

        # --- 1. Write MTZ Files (Unchanged) ---
        for i, rasu in enumerate(dm.asu_collection.reciprocal_asus):
            hkls = rasu.lookup_table.get_hkls().astype(np.int32)
            h, k, l = hkls.T

            # Map sparse HKLs to dense grid indices
            h_idx = np.mod(h, nx)
            k_idx = np.mod(k, ny)
            l_idx = np.mod(l, nz)

            # Since F_grid is now the FULL grid (from _expand_to_full or rfft3d output logic),
            # we can just look up indices directly.
            # Note: rfft3d output (careleast_real) is half-grid,
            # while _expand_to_full (careleast_spectral) is full-grid.
            # We need to handle this distinction or ensure F_grid is consistent.

            # Updated MTZ extraction for Spectral mode using half-grid
            if parser.algorithm in ['careleast_real', 'careleast_spectral']:
                # Both now provide half-grid F_grid (real uses rfft3d output, spectral uses state directly)
                is_friedel = l > (nz // 2)
                idx_h = np.where(is_friedel, (nx - h) % nx, h)
                idx_k = np.where(is_friedel, (ny - k) % ny, k)
                idx_l = np.where(is_friedel, (nz - l) % nz, l)

                # Check bounds to ensure we don't index out of nz_half
                # (idx_l should technically be <= nz//2 if logic is correct)
                F_values = F_grid[idx_h, idx_k, idx_l]
                F_values = np.where(is_friedel, np.conj(F_values), F_values)
            else:
                # careleast_spectral F_grid is FULL grid (from _expand_to_full)
                F_values = F_grid[h_idx, k_idx, l_idx]

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

        # --- 2. Write MRC Map (New) ---
        mrc_name = parser.output_base + '_density.mrc'
        print(f"Writing {mrc_name}...")

        # Prepare Gemmi Grid
        ccp4 = gemmi.Ccp4Map()
        grid_np = final_density.numpy().astype(np.float32)
        # Gemmi requires Fortran ordering (Fastest: X)
        grid_np = np.asfortranarray(grid_np)
        ccp4.grid = gemmi.FloatGrid(grid_np)

        # Set Cell & Spacegroup (P1 for the raw map)
        uc = dm.asu_collection.reciprocal_asus[0].cell
        ccp4.grid.unit_cell = uc
        ccp4.grid.spacegroup = gemmi.SpaceGroup('P1')

        ccp4.update_ccp4_header()
        ccp4.write_ccp4_map(mrc_name)

        # Also save raw numpy if needed
        np.save(parser.output_base + '_density.npy', grid_np)
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
