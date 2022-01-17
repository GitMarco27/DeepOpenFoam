import tensorflow as tf
import logging
from utils.utils import RunBuilder, load_data
import argparse
import json
import os
from utils.custom_objects import chamfer_distance
import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from utils.PointNetAE import create_pointnet_ae, OrthogonalRegularizer, Sampling
from utils.custom_objects import r_squared
from GitMarco.tf.losses import euclidian_dist_loss

logging.basicConfig(level=logging.INFO)
logging.info(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

try:
    for gpu in tf.config.experimental.list_physical_devices(
            "GPU"): tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12000)])
except Exception as e:
    logging.info(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training, validation and testing of AE and 3D regressor')

    # Required positional argument
    parser.add_argument('--data_path', type=str,
                        default='dataset',
                        help='dataset path')
    parser.add_argument('--results_path', type=str,
                        default='results',
                        help='results path')
    parser.add_argument('--log_path', type=str,
                        default='logs',
                        help='log path')
    parser.add_argument('--k', type=int,
                        default=10,
                        help='encoding size')
    parser.add_argument('--lr', type=float,
                        default=0.0001,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int,
                        default=256,
                        help='batch_size')
    parser.add_argument('--decoder', type=str,
                        default='dense',
                        help='decoder type')
    parser.add_argument('--arch_bool', type=bool,
                        default=True,
                        help='architectural_parameters')
    parser.add_argument('--orto_reg', type=bool,
                        default=False,
                        help='Ortogonality regularization')
    parser.add_argument('--beta', type=int,
                        default=1,
                        help='variatinoal encoder beta')
    parser.add_argument('--start_index', type=int,
                        default=0,
                        help='encoding size')
    parser.add_argument('--clear', type=bool,
                        default=True,
                        help='delete results path if exists')

    args = parser.parse_args()

    normed_geometries, normed_global_variables, scaler_globals, min_y, max_y = load_data(args.data_path)

    train_data, test_data, train_labels, test_labels = train_test_split(
        normed_geometries, normed_global_variables, test_size=0.1, shuffle=True, random_state=22)

    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=0.1, shuffle=True)

    params = OrderedDict(
        lr=[args.lr],
        batch_size=[args.batch_size],
        epochs=[20000, ],
        optimizer=['Adam'],
        type_decoder=[args.decoder],
        architectural_parameters=[[args.arch_bool, args.beta]],
        # [is_variational, beta],
        encoding_size=[args.k, ],
        ort_reg_bools=[
            [args.orto_reg, args.orto_reg],
        ],  # [feature_transform, orto_reg]
        reg_drop_out_value=[0., ],
        loss_function=[chamfer_distance]
    )

    # handle_results_path()
    run_count = args.start_index
    run_data = []

    print('train data', train_data.shape)
    print('validation data', val_data.shape)
    print('test data', test_data.shape)

    for run in tqdm.tqdm(RunBuilder.get_runs(params)):

        run_count += 1
        print('\n--- New run detected ---')

        for key in run._asdict():
            print(f'{key}: {run._asdict()[key]}')

        run_path = os.path.join(args.results_path, str(run_count))

        print(f'\n Creating results path: {run_path}')
        os.mkdir(run_path)

        log_dir = os.path.join(args.log_path, f'{args.k}_{run_count}')
        os.mkdir(log_dir)
        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2000)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
            run_path + '/checkpoint', monitor='val_loss', verbose=2, save_best_only=True,
            save_weights_only=False, mode='auto', period=200,
            options=None,
        )

        model = create_pointnet_ae(run,
                                   grid_size=4,
                                   n_geometry_points=400,
                                   n_global_variables=2, )

        model.summary()

        optimizer = tf.keras.optimizers.get(run.optimizer)
        optimizer.learning_rate = run.lr

        model.compile(
            optimizer=optimizer,
            loss=[run.loss_function, tf.keras.metrics.mean_squared_error],
            metrics=dict(reg_gv=[r_squared]),
        )

        with tf.device('gpu:0'):
            history = model.fit(train_data,
                                [train_data, train_labels],
                                batch_size=run.batch_size,
                                epochs=run.epochs,
                                shuffle=True,
                                validation_data=(val_data,
                                                 [val_data, val_labels]),
                                validation_batch_size=run.batch_size,
                                callbacks=[tensorboard_callback,
                                           checkpoints_callback,
                                           earlystop_callback]
                                )

        model.save(os.path.join(run_path, 'model'))

        model = tf.keras.models.load_model(os.path.join(run_path, 'checkpoint'),
                                           custom_objects={'r_squared': r_squared,
                                                           'chamfer_distance': chamfer_distance,
                                                           'euclidian_dist_loss': euclidian_dist_loss,
                                                           'OrthogonalRegularizer': OrthogonalRegularizer,
                                                           'Sampling': Sampling})

        train_scores = model.evaluate(train_data, [train_data, train_labels])
        val_scores = model.evaluate(val_data, [val_data, val_labels])

        results = OrderedDict()
        results["run"] = run_count
        results["train_loss"], results["train_loss_ae"], results["train_loss_reg"] = train_scores[0], train_scores[1], \
                                                                                     train_scores[2]
        results["val_loss"], results["val_loss_ae"], results["val_loss_reg"] = val_scores[0], val_scores[1], \
                                                                               val_scores[2]
        results["train_r2"], results["val_r2"] = train_scores[3], val_scores[3]
        results["run_path"] = run_path

        for k, v in run._asdict().items(): results[k] = v
        run_data.append(results)
        df = pd.DataFrame.from_dict(run_data, orient='columns')

        df.to_excel(os.path.join(args.results_path, f'results_{args.start_index}.xlsx'))

        # with open(os.path.join(run_path, 'log.json'), 'w', encoding='utf-8') as f:
        #     json.dump([results], f, ensure_ascii=False, indent=4)

    print('\n--- Final Df ---\n')
    print(df)

    # df.sort_values('val_r2', axis=1, inplace=True)
    # best_model = tf.keras.models.load(os.path.join(df.run_path[0], 'model'))
    # best_model.evaluate(test_data, [test_data, test_labels])
    # best_model.save(os.path.join(args.results_path, 'best_model'))
