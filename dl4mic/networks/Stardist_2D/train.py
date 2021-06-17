import sys
import os
import time
import math
import warnings
import shutil
import csv
import tensorflow as tf
import glob
from csbdeep.utils import normalize
sys.path.append(r'../lib/')
from dl4mic.cliparser import ParserCreator
from dl4mic.stardist import *

def main(argv):
    parser = ParserCreator.createArgumentParser("./train.yml")
    if len(argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args(argv[1:])
    print(args)

    if tf.test.gpu_device_name() == '':
        print('You do not have GPU access.')
        print('Expect slow performance.')
    else:
        print('You have GPU access')

    print('Tensorflow version is ' + str(tf.__version__))


    X = sorted(glob.glob(args.dataSourcePath+"*.tif"))
    Y = sorted(glob.glob(args.dataTargetPath+"*.tif"))

    X = list(map(imread, X))
    Y = list(map(imread, Y))
    # Normalisation
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
    if n_channel == 1:
        axis_norm = (0,1)   # normalize channels independently
    if n_channel > 1:
        axis_norm = (0,1,2) # normalize channels jointly


    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in X]
    Y = [fill_label_holes(y) for y in Y]
    #Y = [y[:,:,0] for y in Y]

    # Split training and validation
    validationFraction = args.validationFraction

    assert len(X) > 1, "not enough training data"

    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))

    n_val = max(1, int(round((validationFraction/100) * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]

    X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]

    full_model_path = os.path.join(args.baseDir, args.name)
    if os.path.exists(full_model_path):
        print('!! WARNING: Folder already exists and will be overwritten !!')

    conf = Config2D(
        n_rays=args.nRays,
        train_batch_size=args.batchSize,
        n_channel_in=n_channel,
        train_patch_size=(args.patchSizeXY, args.patchSizeXY),
        grid=(args.gridParameter, args.gridParameter),
        train_learning_rate=args.learningRate,
    )

    model = stardist(conf, name=args.name, basedir=args.baseDir)

    if args.resumeTraining:
        model.load_weights(args.startingWeigth)

    random_choice = random.choice(os.listdir(args.dataSourcePath))
    x = imread(args.dataSourcePath + "/" + random_choice)
    Image_Y = x.shape[0]
    Image_X = x.shape[1]

    if args.stepsPerEpoch == 0:
        number_of_steps = int(Image_X * Image_Y / (args.patchSizeXY * args.patchSizeXY) * args.augmentationFactor * (int(len(X) / args.batchSize)))
    else:
        number_of_steps = args.stepsPerEpoch

    augment = None
    if args.augmentationFactor > 1:
        augment = augmenter
        print("Data Augmentation Enabled")

    if os.path.exists(full_model_path):
        print('!! WARNING: Model folder already existed and has been removed !!')
        shutil.rmtree(full_model_path)

    os.makedirs(full_model_path)
    os.makedirs(os.path.join(full_model_path, 'Quality Control'))

    model.keras_model.summary();

    # ------------------ Display ------------------
    print('---------------------------- Main training parameters ----------------------------')
    print('Number of epochs: '+str(args.epochs))
    print('Batch size: '+str(args.batchSize))
    print('Number of training dataset: '+str(len(X)))
    print('Number of training steps: '+str(number_of_steps-n_val))
    print('Number of validation steps: '+str(n_val))
    print('---------------------------- ------------------------ ----------------------------')

    start = time.time()
    
    history = model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augment,
                          epochs=args.epochs,
                          steps_per_epoch=number_of_steps)

    # Save the last model
    #model.keras_model.save(os.path.join(full_model_path, 'weights_last.hdf5'))

    lossDataCSVPath = os.path.join(full_model_path, 'Quality Control/training_evaluation.csv')

    with open(lossDataCSVPath, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history.history.keys())
        values = list(history.history.values())
        #writer.writerows(values)
        for i in range(args.epochs):
            v=[values[j][i] for j in range(len(values))]
            writer.writerow(v)

    print("------------------------------------------")
    dt = time.time() - start
    mins, sec = divmod(dt, 60)
    hour, mins = divmod(mins, 60)
    print("Time elapsed:", hour, "hour(s)", mins, "min(s)", round(sec), "sec(s)")
    print("------------------------------------------")

    print("---training done---")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main(sys.argv)
