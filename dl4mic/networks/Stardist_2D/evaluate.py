import sys
import os
import shutil
import csv
import io
sys.path.append(r'../lib/')
from dl4mic.cliparser import ParserCreator
from dl4mic.stardist import *

def main(argv):
    parser = ParserCreator.createArgumentParser("./evaluate.yml")
    args = parser.parse_args(argv[1:])
    full_QC_model_path = os.path.join(args.baseDir, args.name)
    if os.path.exists(os.path.join(full_QC_model_path, 'weights_best.h5')):
        print("The " + args.name + " network will be evaluated")
    else:
        print('!! WARNING: The chosen model does not exist !!')
        print('Please make sure you provide a valid model path and model name before proceeding further.')

    # Create a quality control/Prediction Folder
    prediction_QC_folder = os.path.join(full_QC_model_path, 'Quality Control', 'Prediction')
    if os.path.exists(prediction_QC_folder):
        shutil.rmtree(prediction_QC_folder)

    os.makedirs(prediction_QC_folder)

    # Load the model

    conf = Config2D(
        n_rays=args.nRays,
        grid=(args.gridParameter,args.gridParameter),
    )
    model = stardist(conf,name=args.name, basedir=args.baseDir)

    model.keras_model.summary();
    model.keras_model.load_weights(os.path.join(full_QC_model_path, 'weights_best.h5'))

    # Create a list of sources
    source_dir_list = os.listdir(args.testInputPath)
    number_of_dataset = len(source_dir_list)
    print('Number of dataset found in the folder: '+str(number_of_dataset))
    im=imread(args.testInputPath +"/"+ source_dir_list[0])
    n_channel = 1 if im.ndim == 2 else im.shape[-1]
    if n_channel == 1:
        axis_norm = (0, 1)  # normalize channels independently
    if n_channel > 1:
        axis_norm = (0, 1, 2)  # normalize channels jointly

    predictions = []
    polygons= []
    for i in range(number_of_dataset):
        print("processing dataset " + str(i+1) + ", file: " + source_dir_list[i])
        image = imread(args.testInputPath +"/"+ source_dir_list[i])
        image = normalize(image,1,99.8, axis=axis_norm)
        labels, poly =model.predict_instances(image)
        predictions.append(labels)
        polygons.append(poly)

    # Save the results in the folder
    saveResult(prediction_QC_folder, predictions,polygons, source_dir_list, prefix=prediction_prefix)

    with open(os.path.join(full_QC_model_path, 'Quality Control', 'QC_metrics_' + args.name + '.csv'), "w",
              newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["File name", "IoU", "False Positives", "True Positives","False Negatives","precision","recall","Accuracy","F1","N True","N Pred","Mean True Score","Mean Matched Score","Panoptic Quality"]);
        
        # Corresponding objects of ground truth and prediction are counted as true positives, false positive, and false negatives whether their intersection over union (IoU) >= thresh
        # mean_true_score is the mean IoUs of matched true positives but normalized by the total number of GT objects
        # mean_matched_score is the mean IoUs of matched true positives
        # panoptic_quality defined as in Eq. 1 of Kirillov et al. "Panoptic Segmentation", CVPR 2019

        # Initialise the lists

        for filename in os.listdir(args.testInputPath):
            if not os.path.isdir(os.path.join(args.testInputPath, filename)):
                print('Running QC on: ' + filename)
                test_input = io.imread(os.path.join(args.testInputPath, filename))
                test_prediction = io.imread(os.path.join(full_QC_model_path,'Quality Control','Prediction', filename))
                test_ground_truth_image = io.imread(os.path.join(args.testGroundTruthPath, filename))

                # Calculate the matching (with IoU threshold `thresh`) and all metrics

                stats = matching(test_ground_truth_image, test_prediction, thresh=0.5)

                # Convert pixel values to 0 or 255
                test_prediction_0_to_255 = test_prediction
                test_prediction_0_to_255[test_prediction_0_to_255 > 0] = 255

                # Convert pixel values to 0 or 255
                test_ground_truth_0_to_255 = test_ground_truth_image
                test_ground_truth_0_to_255[test_ground_truth_0_to_255 > 0] = 255
                # Intersection over Union metric

                intersection = np.logical_and(test_ground_truth_image, test_prediction)
                union = np.logical_or(test_ground_truth_image, test_prediction)
                iou_score = np.sum(intersection) / np.sum(union)
                writer.writerow([filename, str(iou_score), str(stats.fp), str(stats.tp), str(stats.fn), str(stats.precision),
                                 str(stats.recall), str(stats.accuracy), str(stats.f1), str(stats.n_true),
                                 str(stats.n_pred), str(stats.mean_true_score), str(stats.mean_matched_score),
                                 str(stats.panoptic_quality)])


    print("---evaluation done---")


if __name__ == '__main__':
    main(sys.argv)
