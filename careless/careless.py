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
        message="Training",
        validation_data=test,
        validation_frequency=validation_frequency,
        progress=progress,
        reduce_retracing=parser.reduce_retracing,
        jit_compile=parser.jit_compile,
    )

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

