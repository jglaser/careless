import numpy as np
import tensorflow as tf
import tf_keras as tfk
import reciprocalspaceship as rs
import gemmi
from .asu import ReciprocalASU, ReciprocalASUCollection
from careless.models.base import BaseModel
from careless.models.priors.wilson import WilsonPrior, DoubleWilsonPrior


class pSGLD(tfk.optimizers.Optimizer):
    """
    Stochastic Gradient Langevin Dynamics with RMSProp preconditioning.
    Compatible with Keras Eager Execution.
    """
    def __init__(self, learning_rate=0.001, precondition_decay=0.95, epsilon=1e-7, name="pSGLD", **kwargs):
        super().__init__(name=name, **kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.precondition_decay = precondition_decay
        self.epsilon = epsilon

    def build(self, var_list):
        # 1. Initialize Parent (Keras) structures first
        super().build(var_list)

        if hasattr(self, "_built") and self._built:
            return

        # 2. Initialize our custom preconditioners
        self.preconditioners = []
        self._var_to_index = {} # Use a custom map name to avoid conflict

        for i, var in enumerate(var_list):
            self.preconditioners.append(self.add_variable_from_reference(var, "preconditioner"))
            # Use var.ref() as the key for reliable lookup in Eager mode
            self._var_to_index[var.ref()] = i

        self._built = True

    def update_step(self, gradient, variable):
        lr = tf.cast(self.learning_rate, variable.dtype)
        if isinstance(self.learning_rate, tfk.optimizers.schedules.LearningRateSchedule):
            lr = self.learning_rate(self.iterations)
            lr = tf.cast(lr, variable.dtype)

        # Lookup our custom index
        if variable.ref() not in self._var_to_index:
            return # Should not happen if build() was called correctly

        idx = self._var_to_index[variable.ref()]
        preconditioner = self.preconditioners[idx]

        # 1. Update Preconditioner (RMSProp style)
        new_pre = self.precondition_decay * preconditioner + (1.0 - self.precondition_decay) * tf.square(gradient)
        preconditioner.assign(new_pre)

        # 2. Compute Preconditioned Gradient
        pre_sqrt = tf.sqrt(new_pre + self.epsilon)

        # 3. Langevin Update
        # theta_new = theta - lr * grad / M + noise
        scaled_grad = gradient / pre_sqrt
        noise_std = tf.sqrt(2.0 * lr / pre_sqrt)
        noise = tf.random.normal(tf.shape(variable), stddev=noise_std, dtype=variable.dtype)

        new_var = variable - lr * scaled_grad + noise
        variable.assign(new_var)

class DataManager():
    """
    This class comprises various data manipulation methods as well as methods to aid in model construction.
    """
    parser = None
    def __init__(self, inputs, asu_collection, parser=None):
        """
        Parameters
        ----------
        inputs : tuple
        asu_collection : ReciprocalASUCollection
        parser : Namespace (optional)
            A Namespace instance created by careless.parser.parser.parse_args()
        """
        self.inputs = inputs
        self.asu_collection = asu_collection
        self.parser = parser
        
        # Determine if we are handling complex structure factors
        self.is_complex = False
        if self.parser is not None:
            if hasattr(self.parser, 'surrogate_posterior'):
                if self.parser.surrogate_posterior in ('complex_flow', 'real_space_grid_flow'):
                    self.is_complex = True
            # Also set complex if using SGLD algorithm
            if hasattr(self.parser, 'algorithm') and self.parser.algorithm == 'sgld':
                self.is_complex = True

    @classmethod
    def from_pickle(cls, filename):
        import pickle
        with open(filename, 'rb') as f:
            dm = pickle.load(f)
        return dm

    @classmethod
    def from_mtz_files(cls, filenames, formatter):
        return cls.from_datasets((rs.read_mtz(i) for i in filenames), formatter)

    @classmethod
    def from_stream_files(cls, filenames, formatter):
        return cls.from_datasets((rs.read_crystfel(i) for i in filenames), formatter)

    @staticmethod
    def wilson_sigma(b, dHKL):
        sigma = np.exp(-0.25 * b * np.reciprocal(dHKL*dHKL))
        return sigma

    def get_wilson_sigma(self, b=None):
        if b is None:
            return 1.
        sigma = self.wilson_sigma(b, self.asu_collection.dHKL)
        return sigma

    def get_wilson_prior(self, b=None, k=1.):
        """ Construct a wilson prior with an optional temperature factor, b, appropriate for self.asu_collection. """
        if b is None:
            sigma = 1.
        elif isinstance(b, float):
            sigma = self.get_wilson_sigma(b)
        else:
            raise ValueError(f"parameter b has type{type(b)} but float was expected")
        sigma = sigma * k

        return WilsonPrior(
            self.asu_collection.centric,
            self.asu_collection.multiplicity,
            sigma,
        )

    def get_tf_dataset(self, inputs=None):
        if inputs is None:
            inputs = self.inputs

        inputs = tuple(inputs)
        iobs = BaseModel.get_intensities(inputs)
        sigiobs = BaseModel.get_uncertainties(inputs)
        packed  = (inputs, iobs, sigiobs)
        tfds = tf.data.Dataset.from_tensor_slices(packed)
        return tfds.batch(len(iobs))

    def get_predictions(self, model, inputs=None, test_value=0):
        laue = BaseModel.is_laue(inputs)

        if inputs is None:
            inputs = self.inputs

        refl_id = BaseModel.get_refl_id(inputs)
        asu_id,H = self.asu_collection.to_asu_id_and_miller_index(refl_id)
        asu_id = asu_id.flatten()
        file_id = model.get_file_id(inputs).flatten()
        image_id = model.get_image_id(inputs).flatten()
        if laue:
            harmonic_id = BaseModel.get_harmonic_id(inputs).flatten()
        else:
            harmonic_id = np.arange(len(refl_id))
        h,k,l = H.T

        output = rs.DataSet({
            'H' : rs.DataSeries(h, dtype='H'),
            'K' : rs.DataSeries(k, dtype='H'),
            'L' : rs.DataSeries(l, dtype='H'),
            'harmonic_id' : rs.DataSeries(harmonic_id, dtype='I'),
            'asu_id'      : rs.DataSeries(asu_id, dtype='I'),
            'image_id'    : rs.DataSeries(image_id, dtype='I'),
            'file_id'     : rs.DataSeries(file_id, dtype='I'),
            'test'        : rs.DataSeries(test_value * np.ones_like(h), dtype='I'),
        }, merged=False)
        _,idx = np.unique(output.harmonic_id, return_index=True)
        output = output.loc[idx].reset_index(drop=True)
        del(output['harmonic_id'])

        iobs = BaseModel.get_intensities(inputs).flatten()
        sig_iobs = BaseModel.get_uncertainties(inputs).flatten()
        ipred,sigipred = model.prediction_mean_stddev(inputs)
        scale,sigscale = model.scale_mean_stddev(inputs)

        num_refls = len(output)
        data_cols = {
            'Iobs'    : rs.DataSeries(iobs[:num_refls], dtype='J'),
            'SigIobs' : rs.DataSeries(sig_iobs[:num_refls], dtype='Q'),
            'Ipred'   : rs.DataSeries(ipred[:num_refls], dtype='J'),
            'SigIpred': rs.DataSeries(sigipred[:num_refls], dtype='Q'),
            'Scale'   : rs.DataSeries(scale[:num_refls], dtype='J'),
            'SigScale': rs.DataSeries(sigscale[:num_refls], dtype='Q'),
        }
        for k,v in data_cols.items():
            output[k] = v

        for i,rasu in enumerate(self.asu_collection):
            idx = output['asu_id'] == i
            result = output.loc[idx]
            result.cell = rasu.cell
            result.spacegroup = rasu.spacegroup
            yield result.set_index(['H', 'K', 'L'])


    def get_results(self, surrogate_posterior, inputs=None, output_parameters=True, max_intensity_snr=1e-5):
        if inputs is None:
            inputs = self.inputs

        # 1. Retrieve Statistics based on Complexity Flag
        if self.is_complex:
            # --- Complex Case ---
            F_centroid_complex = surrogate_posterior.mean().numpy()
            samples = surrogate_posterior.sample(200).numpy()
            samples_amp = np.abs(samples)
            mean_amp = np.mean(samples_amp, axis=0) # <|F|>

            abs_centroid = np.abs(F_centroid_complex)
            FOM = np.divide(abs_centroid, mean_amp, out=np.zeros_like(mean_amp), where=mean_amp!=0)
            FOM = np.clip(FOM, 0.0, 1.0) 

            F = mean_amp
            PHI = np.angle(F_centroid_complex, deg=True)

            if hasattr(surrogate_posterior, 'mean_intensity'):
                I = surrogate_posterior.mean_intensity().numpy()
            else:
                I = np.mean(samples_amp**2, axis=0)

            if hasattr(surrogate_posterior, 'moment_4_intensity'):
                f4 = surrogate_posterior.moment_4_intensity().numpy()
            else:
                f4 = np.mean(samples_amp**4, axis=0)

            SigF = surrogate_posterior.stddev().numpy()

        else:
            # --- Standard Case ---
            F = surrogate_posterior.mean().numpy()
            SigF = surrogate_posterior.stddev().numpy()
            I = SigF * SigF + F * F

            try:
                f4 = surrogate_posterior.moment_4(method='scipy')
            except:
                 f4 = surrogate_posterior.moment_4(n_samples=100).numpy()

        ivar = np.square(I * max_intensity_snr)
        ivar = np.maximum(ivar, f4 - I * I)
        SigI = np.sqrt(ivar)

        params = None
        if output_parameters:
            params = {}
            for k in sorted(surrogate_posterior.parameter_properties()):
                v = surrogate_posterior.parameters[k]
                numpify = lambda x : tf.convert_to_tensor(x).numpy()
                val = numpify(v).flatten()
                if len(val) == 1:
                    val = val * np.ones(len(F), dtype='float32')
                params[k] = val

        asu_id,H = self.asu_collection.to_asu_id_and_miller_index(np.arange(len(F)))
        h,k,l = H.T
        refl_id = BaseModel.get_refl_id(inputs)
        N = np.bincount(refl_id.flatten(), minlength=len(F)).astype('float32')
        results = ()
        for i,asu in enumerate(self.asu_collection):
            idx = (asu_id == i).flatten()

            data_dict = {
                'H' : h[idx], 'K' : k[idx], 'L' : l[idx],
                'F' : F[idx], 'SigF' : SigF[idx],
                'I' : I[idx], 'SigI' : SigI[idx],
                'N' : N[idx],
            }

            if self.is_complex:
                data_dict['PHI'] = rs.DataSeries(PHI[idx], dtype='P')
                data_dict['FOM'] = rs.DataSeries(FOM[idx], dtype='W')

            output = rs.DataSet(
                data_dict,
                cell=asu.cell,
                spacegroup=asu.spacegroup,
                merged=True,
            ).infer_mtz_dtypes().set_index(['H', 'K', 'L'])

            if params is not None:
                for key in sorted(params.keys()):
                    val = params[key]
                    output[key] = rs.DataSeries(val[idx], index=output.index, dtype='R')

            output = output[output.N > 0]

            if asu.anomalous:
                output = output.unstack_anomalous()
                anom_keys = [
                    'F(+)', 'SigF(+)', 'F(-)', 'SigF(-)',
                    'I(+)', 'SigI(+)', 'I(-)', 'SigI(-)',
                    'N(+)', 'N(-)'
                ]
                if self.is_complex:
                    anom_keys += ['PHI(+)', 'PHI(-)', 'FOM(+)', 'FOM(-)']

                reorder = [k for k in anom_keys if k in output] + [key for key in output if key not in anom_keys]
                output = output[reorder]

            results += (output, )
        return results

    # <-- start xval data splitting methods
    def split_mono_data_by_mask(self, test_idx):
        test,train = (),()
        for inp in self.inputs:
            test  += (inp[ test_idx.flatten(),...] ,)
            train += (inp[~test_idx.flatten(),...] ,)
        return train, test

    def split_data_by_refl(self, test_fraction=0.5):
        if BaseModel.is_laue(self.inputs):
            harmonic_id = BaseModel.get_harmonic_id(self.inputs)
            test_idx = (np.random.random(harmonic_id.max()+1) <= test_fraction)[harmonic_id]
            train, test = self.split_laue_data_by_mask(test_idx)
            return train, test

        test_idx = np.random.random(len(self.inputs[0])) <= test_fraction
        train, test = self.split_mono_data_by_mask(test_idx)
        return train, test

    def split_laue_data_by_mask(self, test_idx):
        harmonic_id = BaseModel.get_harmonic_id(self.inputs)

        isect = np.intersect1d(
            harmonic_id[test_idx].flatten(),
            harmonic_id[~test_idx].flatten(),
        )
        if len(isect) > 0:
            raise ValueError(f"test_idx splits harmonic observations with harmonic_id : {isect}")

        def split(inputs, idx):
            harmonic_id = BaseModel.get_harmonic_id(inputs)
            result = ()
            uni,inv = np.unique(harmonic_id[idx], return_inverse=True)
            for i,v in enumerate(inputs):
                name = BaseModel.get_name_by_index(i)
                if name in ('intensities', 'uncertainties'):
                    v = v[uni]
                    v = np.pad(v, [[0, len(inv) - len(v)], [0, 0]], constant_values=1.)
                elif name == 'harmonic_id':
                    v = inv[:,None]
                else:
                    v = v[idx.flatten(),...]
                result += (v ,)
            return result

        return split(self.inputs, ~test_idx), split(self.inputs, test_idx)

    def split_data_by_image(self, test_fraction=0.5):
        image_id = BaseModel.get_image_id(self.inputs)
        test_idx = np.random.random(image_id.max()+1) <= test_fraction

        if True not in test_idx:
            test_idx[0] = True
        elif False not in test_idx:
            test_idx[0] = False
            
        test_idx = test_idx[image_id]
        if BaseModel.is_laue(self.inputs):
            train, test = self.split_laue_data_by_mask(test_idx)
        else:
            train, test = self.split_mono_data_by_mask(test_idx)
        return train, test
    # --> end xval data splitting methods

    def get_likelihood(self, parser=None):
        if parser is None:
            parser = self.parser
        
        if parser.type == 'poly':
            if parser.refine_uncertainties:
                from careless.models.likelihoods.laue import NormalEv11Likelihood as NormalLikelihood
                from careless.models.likelihoods.laue import StudentTEv11Likelihood as StudentTLikelihood
            else:
                from careless.models.likelihoods.laue import NormalLikelihood, StudentTLikelihood
        elif parser.type == 'mono':
            if parser.refine_uncertainties:
                from careless.models.likelihoods.mono import NormalEv11Likelihood as NormalLikelihood
                from careless.models.likelihoods.mono import StudentTEv11Likelihood as StudentTLikelihood
            else:
                from careless.models.likelihoods.mono import NormalLikelihood, StudentTLikelihood

        dof = parser.studentt_likelihood_dof
        if dof is None:
            likelihood = NormalLikelihood()
        else:
            likelihood = StudentTLikelihood(dof)
        return likelihood

    def get_scaling_model(self, parser=None):
        from careless.models.scaling.image import HybridImageScaler, ImageScaler, NeuralImageScaler
        from careless.models.scaling.nn import MLPScaler
        from careless.models.scaling.spectral import TabulatedSpectralScaler
        
        if parser is None:
            parser = self.parser

        mlp_width = parser.mlp_width
        if mlp_width is None:
            mlp_width = BaseModel.get_metadata(self.inputs).shape[-1]

        if parser.scale_bijector.lower() == 'softplus':
            from tensorflow_probability import bijectors as tfb
            scale_bijector = tfb.Chain([
                tfb.Shift(parser.epsilon),
                tfb.Softplus(),
            ])
            istd = BaseModel.get_intensities(self.inputs).std()
        elif parser.scale_bijector.lower() == 'exp':
            from tensorflow_probability import bijectors as tfb
            scale_bijector = tfb.Chain([
                tfb.Shift(parser.epsilon),
                tfb.Exp(),
            ])
            istd = None
        else:
            raise ValueError(f"Unsupported scale bijector type, {parser.scale_bijector}")

        if parser.spectral_file is not None:
            data = np.loadtxt(parser.spectral_file)
            x_grid = data[:, 0]
            y_grid = data[:, 1]
            scaling_model = TabulatedSpectralScaler(
                x_grid=x_grid,
                y_grid=y_grid,
                trainable_scale=parser.trainable_spectral_scale,
                num_grid_points=parser.spectral_grid_points,
                lorentz_correction=parser.lorentz_correction,
            )
        elif parser.image_layers > 0:
            n_images = np.max(BaseModel.get_image_id(self.inputs)) + 1
            scaling_model = NeuralImageScaler(
                parser.image_layers,
                n_images,
                parser.mlp_layers,
                mlp_width,
                epsilon=parser.epsilon,
                scale_bijector=scale_bijector,
                scale_multiplier=istd,
            )
        else:
            mlp_scaler = MLPScaler(
                parser.mlp_layers, mlp_width, 
                epsilon=parser.epsilon, scale_bijector=scale_bijector, scale_multiplier=istd,
            )
            if parser.use_image_scales:
                n_images = np.max(BaseModel.get_image_id(self.inputs)) + 1
                image_scaler = ImageScaler(n_images)
                scaling_model = HybridImageScaler(mlp_scaler, image_scaler)
            else:
                scaling_model = mlp_scaler
        
        return scaling_model

    def build_model(self, parser=None, surrogate_posterior=None, prior=None, likelihood=None, scaling_model=None, mc_sample_size=None):
        """
        Build the model specified in parser.
        Supports both Variational Inference (VI) and SGLD algorithms.
        """
        from careless.models.merging.surrogate_posteriors import TruncatedNormal, FlowPosterior
        from careless.models.merging.variational import VariationalMergingModel
        import tf_keras as tfk
        import tensorflow_probability as tfp
        import gemmi
        
        if parser is None:
            parser = self.parser
        if parser is None:
            raise ValueError("No parser supplied, but self.parser is unset")

        # --- 1. Setup Likelihood and Scaling (Shared) ---
        if likelihood is None:
            likelihood = self.get_likelihood(parser)
        if scaling_model is None:
            scaling_model = self.get_scaling_model(parser)

        # --- 2. Setup Base Prior (Reciprocal Space / Wilson) ---
        if prior is None:
            parents = parser.parents
            r_values = parser.dwr
            if parents is None:
                wilson_prior = self.get_wilson_prior(parser.wilson_prior_b)
            else:
                parents = [None if i == 'None' else int(i) for i in parents.split(',')]
                r_values = [float(i) for i in r_values.split(',')]
                sigma = self.get_wilson_sigma(parser.wilson_prior_b)
                reindexing_ops = parser.reindexing_ops
                if reindexing_ops is not None:
                    delim = ';'
                    reindexing_ops = [gemmi.Op(i) for i in reindexing_ops.split(delim)]
                wilson_prior = DoubleWilsonPrior(self.asu_collection, parents, r_values, reindexing_ops, sigma=sigma, optimize_r=parser.optimize_double_wilson_r)
            
            prior = wilson_prior

        # --- 3. Grid Size Auto-Detection ---
        grid_shape = getattr(parser, 'grid_shape', None)
        
        # OLD LINE:
        # needs_grid = (getattr(parser, 'algorithm', 'vi') == 'sgld') or \
        #              (getattr(parser, 'surrogate_posterior', '') == 'real_space_grid_flow')

        # NEW LINE: Auto-detect grid for ANY complex model (enables Dense Prior for complex_flow)
        needs_grid = self.is_complex

        if needs_grid and grid_shape is None:
            d_min = self.asu_collection.dHKL.min()
            cell = self.asu_collection.reciprocal_asus[0].cell
            target_spacing = d_min / 2.0 
            nx = int(cell.a / target_spacing)
            ny = int(cell.b / target_spacing)
            nz = int(cell.c / target_spacing)
            grid_shape = (nx, ny, nz)
            print(f"Auto-detected FFT Grid Shape for d_min={d_min:.2f}A: {grid_shape}")

        # --- 4. Algorithm Branching ---
        
        # === A. SGLD Algorithm ===
        if getattr(parser, 'algorithm', 'vi') == 'sgld':
            print(f"Initializing SGLD Model with Dense Real-Space Prior (TV={getattr(parser, 'tv_weight', 0.0)})...")
            from careless.models.merging.surrogate_posteriors import ParticleSurrogate
            from careless.models.priors.empirical import DenseRealSpacePrior
            from careless.models.priors.base import JointPrior
            from careless.models.merging.variational import JointVariationalMergingModel

            if grid_shape is None:
                raise ValueError("Could not estimate grid shape for SGLD.")

            # 1. Surrogate: Particles (Centered at 0, Correct Variance)
            if surrogate_posterior is None:
                init_scale = tf.sqrt(prior.sigma / 2.0)
                surrogate_posterior = ParticleSurrogate(
                    n_refls=len(prior.mean()),
                    n_particles=parser.mc_samples,
                    initial_loc=None, 
                    initial_scale=init_scale,
                    name='structure_factor'
                )

            # 2. Prior: Joint (Wilson + Dense Real-Space)
            if not isinstance(prior, JointPrior):
                dense_prior = DenseRealSpacePrior(
                    asu_collection=self.asu_collection,
                    grid_size=tuple(grid_shape),
                    tv_weight=getattr(parser, 'tv_weight', 0.0),
                    positivity_weight=getattr(parser, 'positivity_weight', 0.0),
                    sparsity_weight=getattr(parser, 'sparsity_weight', 0.0)
                )
                prior = JointPrior(wilson_prior=prior, sparse_prior=dense_prior)

            model = JointVariationalMergingModel(
                surrogate_posterior, prior, likelihood, scaling_model, 
                parser.mc_samples, kl_weight=parser.kl_weight
            )

            # 3. Optimizer: Custom Keras pSGLD
            lr_schedule = tfk.optimizers.schedules.ExponentialDecay(
                parser.learning_rate,
                decay_steps=parser.iterations // 4,
                decay_rate=0.5,
                staircase=True
            )
            
            # Use local pSGLD class instead of TFP
            opt = pSGLD(
                learning_rate=lr_schedule,
                precondition_decay=0.95,
                name="pSGLD"
            )
            
            model.compile(opt, run_eagerly=parser.run_eagerly)
            return model


        # === B. Variational Inference (VI) - Default ===
        elif self.is_complex:
            # --- Complex Flow Mode ---
            from careless.models.merging.surrogate_posteriors import ComplexCartesianFlow, RealSpaceGridFlow
            from careless.models.priors.base import JointPrior
            from careless.models.priors.empirical import DenseRealSpacePrior, SparseRealSpacePrior # Import Both
            from careless.models.merging.variational import JointVariationalMergingModel

            print("Initializing Complex VI Model...")

            # 1. Surrogate
            if surrogate_posterior is None:
                if getattr(parser, 'surrogate_posterior', '') == 'real_space_grid_flow':
                    if grid_shape is None:
                        raise ValueError("RealSpaceGridFlow requires grid shape.")
                    
                    hkls = self.asu_collection.reciprocal_asus[0].lookup_table.get_hkls()
                    surrogate_posterior = RealSpaceGridFlow(
                        miller_indices=hkls,
                        grid_shape=tuple(grid_shape),
                        depth=parser.flow_depth,
                        filters=getattr(parser, 'cnn_filters', 32),
                        inference_samples=parser.flow_inference_samples,
                        name='structure_factor'
                    )
                else:
                    # Default: ComplexCartesianFlow
                    base_loc = prior.mean()
                    base_scale = tf.sqrt(prior.stddev() / 2.0)
                    
                    surrogate_posterior = ComplexCartesianFlow(
                        loc=base_loc,
                        scale=base_scale,
                        depth=parser.flow_depth,
                        hidden_units=parser.flow_hidden_units,
                        inference_samples=parser.flow_inference_samples,
                        name='structure_factor'
                    )

            # 2. Prior (UPGRADED)
            if not isinstance(prior, JointPrior):
                # Check if we can use the Dense Prior (Better gradients, enforcing symmetry)
                if grid_shape is not None:
                    print(f"Using Dense Real-Space Prior on grid {grid_shape} (TV={getattr(parser, 'tv_weight', 0.0)})")
                    real_space_prior = DenseRealSpacePrior(
                        asu_collection=self.asu_collection,
                        grid_size=tuple(grid_shape),
                        tv_weight=getattr(parser, 'tv_weight', 0.0),
                        positivity_weight=getattr(parser, 'positivity_weight', 0.0),
                        sparsity_weight=getattr(parser, 'sparsity_weight', 0.0)
                    )
                else:
                    # Fallback to Stochastic if grid not detected (shouldn't happen with auto-detect)
                    print("Using Stochastic Sparse Prior")
                    hkls = self.asu_collection.reciprocal_asus[0].lookup_table.get_hkls()
                    real_space_prior = SparseRealSpacePrior(
                        miller_indices=hkls,
                        n_points=getattr(parser, 'stochastic_points', 4096),
                        positivity_weight=getattr(parser, 'positivity_weight', 1.0),
                        sparsity_weight=getattr(parser, 'sparsity_weight', 0.1),
                        tv_weight=getattr(parser, 'tv_weight', 0.0)
                    )
                
                prior = JointPrior(wilson_prior=prior, sparse_prior=real_space_prior)

            # 3. Model
            model = JointVariationalMergingModel(
                surrogate_posterior, prior, likelihood, scaling_model, 
                parser.mc_samples, kl_weight=parser.kl_weight
            )
        else:
            # Standard Real Mode
            if surrogate_posterior is None:
                loc, scale = prior.mean(), prior.stddev()
                scale = scale * parser.structure_factor_init_scale
                low = (1e-32 * ~self.asu_collection.centric).astype('float32')
                
                if parser.surrogate_posterior == 'flow':
                     surrogate_posterior = FlowPosterior.from_loc_and_scale(
                        loc, scale, 
                        depth=parser.flow_depth, 
                        hidden_units=parser.flow_hidden_units,
                        inference_samples=parser.flow_inference_samples,
                        name='structure_factor'
                     )
                else:
                    surrogate_posterior = TruncatedNormal.from_loc_and_scale(loc, scale, low, scale_shift=parser.epsilon)

            model = VariationalMergingModel(
                surrogate_posterior, prior, likelihood, scaling_model, 
                parser.mc_samples, kl_weight=parser.kl_weight
            )

        opt = tfk.optimizers.Adam(
            parser.learning_rate,
            parser.beta_1,
            parser.beta_2,
            clipnorm=parser.clipnorm,
            clipvalue=parser.clipvalue,
            global_clipnorm=parser.global_clipnorm,
        )

        model.compile(opt, run_eagerly=parser.run_eagerly)
        return model
