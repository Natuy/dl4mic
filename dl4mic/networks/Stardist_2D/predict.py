import sys
import os
import numpy as np
from io import StringIO
sys.stderr = StringIO()
sys.path.append(r'../lib/')
from dl4mic.cliparser import ParserCreator
from dl4mic.stardist import *
sys.stderr = sys.__stderr__

def main(argv):
    prediction_prefix = ""
    parser = ParserCreator.createArgumentParser("./predict.yml")
    args = parser.parse_args(argv[1:])
    full_Prediction_model_path = os.path.join(args.baseDir, args.name)
    if os.path.exists(os.path.join(full_Prediction_model_path, 'weights_best.h5')):
        print("The " + args.name + " network will be used.")
    else:
        print('!! WARNING: The chosen model does not exist !!')
        print('Please make sure you provide a valid model path and model name before proceeding further.')

    conf = Config2D(
        n_rays=args.nRays,
        grid=(2,2),
    )
    model = stardist(conf,name=args.name, basedir=args.baseDir)

    model.keras_model.summary();
    model.keras_model.load_weights(os.path.join(full_Prediction_model_path, 'weights_best.h5'))

    source_dir_list = os.listdir(args.dataPath)
    number_of_dataset = len(source_dir_list)
    print('Number of dataset found in the folder: ' + str(number_of_dataset))

    im=imread(args.dataPath +"/"+ source_dir_list[0])
    n_channel = 1 if im.ndim == 2 else im.shape[-1]
    if n_channel == 1:
        axis_norm = (0, 1)  # normalize channels independently
    if n_channel > 1:
        axis_norm = (0, 1, 2)  # normalize channels jointly

    predictions = []
    polygons= []
    for i in range(number_of_dataset):
        print("processing dataset " + str(i+1) + ", file: " + source_dir_list[i])
        image = imread(args.dataPath +"/"+ source_dir_list[i])
        image = normalize(image, 1, 99.8, axis=axis_norm)
        labels, poly =model.predict_instances(image)
        predictions.append(labels)
        polygons.append(poly)

    # Save the results in the folder
    saveResult(args.output, predictions,polygons, source_dir_list, prefix=prediction_prefix)

    print("---predictions done---")


if __name__ == '__main__':
    main(sys.argv)