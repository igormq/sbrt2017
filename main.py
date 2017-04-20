from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Preventing pool_allocator message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import datetime

import tensorflow as tf

import keras
import keras.backend as K
from keras.optimizers import Adam

import preprocessing
from callbacks import MetaCheckpoint, ProgbarLogger
from datasets.dataset_generator import DatasetGenerator
from model import ctc_dummy_loss, decoder_dummy_loss, sbrt2017, ler
from utils.hparams import HParams
import utils.generic_utils as utils
from utils.core_utils import setup_gpu
from utils.core_utils import load_model


def main(args):

    # hack in ProgbarLogger: avoid printing the dummy losses
    keras.callbacks.ProgbarLogger = lambda: ProgbarLogger(
        show_metrics=['loss', 'decoder_ler', 'val_loss', 'val_decoder_ler'])

    # GPU configuration
    setup_gpu(args.gpu, args.allow_growth,
              log_device_placement=args.verbose > 1)

    # Initial configuration
    epoch_offset = 0
    meta = None

    default_args = parser.parse_args([args.mode,
                                      '--dataset', args.dataset,
                                      ])

    args_nondefault = utils.parse_nondefault_args(args,
                                                  default_args)

    if args.mode == 'eval':
        model, meta = load_model(args.load, return_meta=True, mode='eval')

        args = HParams(**meta['training_args']).update(vars(args_nondefault))
        args.mode = 'eval'
    else:
        if args.load:

            print('Loading model...')
            model, meta = load_model(args.load, return_meta=True)

            print('Loading parameters...')
            args = HParams(**meta['training_args']).update(vars(args_nondefault))

            epoch_offset = len(meta['epochs'])
            print('Current epoch: %d' % epoch_offset)

            if args_nondefault.lr:
                print('Setting current learning rate to %f...' % args.lr)
                K.set_value(model.optimizer.lr, args.lr)
        else:
            print('Creating model...')
            # Load model
            model = sbrt2017(num_hiddens=args.num_hiddens,
                             var_dropout=args.var_dropout,
                             dropout=args.dropout,
                             weight_decay=args.weight_decay)

            print('Setting the optimizer...')
            # Optimization
            opt = Adam(lr=args.lr, clipnorm=args.clipnorm)

            # Compile with dummy loss
            model.compile(loss={'ctc': ctc_dummy_loss,
                                'decoder': decoder_dummy_loss},
                          optimizer=opt, metrics={'decoder': ler},
                          loss_weights=[1, 0])

    print('Creating results folder...')
    if args.save is None:
        args.save = os.path.join('results',
                                 'sbrt2017_%s' % (datetime.datetime.now()))
    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    if args.mode == 'train':
        print('Adding callbacks')
        # Callbacks
        model_ckpt = MetaCheckpoint(os.path.join(args.save, 'model.h5'),
                                    training_args=args, meta=meta)
        best_ckpt = MetaCheckpoint(
            os.path.join(args.save, 'best.h5'), monitor='val_decoder_ler',
            save_best_only=True, mode='min', training_args=args, meta=meta)
        callback_list = [model_ckpt, best_ckpt]

    print('Getting the text parser...')
    # Recovering text parser
    label_parser = preprocessing.SimpleCharParser()

    print('Getting the data generator...')
    # Data generator
    data_gen = DatasetGenerator(None, label_parser,
                                batch_size=args.batch_size,
                                seed=args.seed)

    # iterators over datasets
    train_flow, valid_flow, test_flow = None, None, None
    num_val_samples = num_test_samples = 0

    print(str(vars(args)))
    print('Generating flow...')

    if args.mode == 'train':
        train_flow, valid_flow, test_flow = data_gen.flow_from_fname(
            args.dataset, datasets=['train', 'valid', 'test'])
        num_val_samples = valid_flow.len
        print('Initialzing training...')
        # Fit the model
        model.fit_generator(train_flow, samples_per_epoch=train_flow.len,
                            nb_epoch=args.num_epochs, validation_data=valid_flow,
                            nb_val_samples=num_val_samples, max_q_size=10,
                            nb_worker=1, callbacks=callback_list, verbose=1,
                            initial_epoch=epoch_offset)

        del model
        model = load_model(os.path.join(args.save, 'best.h5'), mode='eval')
    else:
        test_flow = data_gen.flow_from_fname(
            args.dataset, datasets='test')

    print('Evaluating model on test set')
    metrics = model.evaluate_generator(test_flow, test_flow.len,
                                       max_q_size=10, nb_worker=1)

    msg = 'Total loss: %.4f\n\
CTC Loss: %.4f\nLER: %.2f%%' % (metrics[0], metrics[1], metrics[3]*100)
    print(msg)

    K.clear_session()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training an ASR system.')

    parser.add_argument('mode', type=str, choices=['train', 'eval'],
                        help='train ou eval mode')

    # Resume training
    parser.add_argument('--load', default=None, type=str)

    # Model settings
    parser.add_argument('--num_hiddens', default=1024, type=int)
    parser.add_argument('--var_dropout', default=0.2, type=float)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)

    # Hyper parameters
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--clipnorm', default=400, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    # End of hyper parameters

    # Dataset definitions
    parser.add_argument('--dataset', default=None, type=str, required='True')

    # Other configs
    parser.add_argument('--save', default=None, type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--allow_growth', default=False, action='store_true')
    parser.add_argument('--verbose', default=0, type=int)
    parser.add_argument('--seed', default=None, type=float)

    args = parser.parse_args()

    args = HParams(**vars(args))

    main(args)
