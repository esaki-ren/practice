import argparse

def get_params():
    parser = argparse.ArgumentParser()

    # stft_params
    stft_args = parser.add_argument_group('STFT parameters')
    stft_args.add_argument('-frame_period', default=5, type=int,
                            help='the length of stft frame')
    stft_args.add_argument('-fs', default=16000, type=int,
                            help='sampling frequency')
    stft_args.add_argument('-nperseg', default=None, type=int,
                            help='nperseg')
    stft_args.add_argument('-window', default='hann', 
                            help='window function.')

    # training_params
    training_args = parser.add_argument_group('training parameters')
    training_args.add_argument('-bn', '--batch_size', default=32, type=int,
                                help='training batch size')
    training_args.add_argument('-epoch', default=5, type=int,
                                help='total epoch')
    training_args.add_argument('-lr', default=1e-3, 
                                help='initial learning rate')
    training_args.add_argument('-wright_decay', default=1e-6, 
                                help='weight decay')

    # model_params
    model_args = parser.add_argument_group('model parameters')
    model_args.add_argument('-in_dim', default=80, type=int,
                            help='input dimension')
    model_args.add_argument('-h_dim1', default=80, type=int,
                            help='hidden layer1 hidden dimension')

    return parser