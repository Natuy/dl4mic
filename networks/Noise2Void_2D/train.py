import sys
import warnings
from io import StringIO
sys.stderr = StringIO()
from dl4mic.n2v import N2VNetwork, ParserCreator
sys.stderr = sys.__stderr__

def main(argv):
    parser = ParserCreator.createArgumentParser("./train.yml")
    if len(argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args(argv[1:])
    print(args)
    N2VNetwork.reportGPUAccess()
    N2VNetwork.reportDevices()
    N2VNetwork.reportTensorFlowAccess()
    n2v = N2VNetwork(args.name)
    n2v.setTrainingSource(args.dataPath)
    n2v.setPath(args.baseDir)
    n2v.setNumberOfEpochs(args.epochs)
    n2v.setPatchSize(args.patchSizeXY)
    n2v.setBatchSize(args.batchSize)
    n2v.setPercentValidation(args.validationFraction)
    n2v.setNumberOfSteps(args.stepsPerEpoch)
    n2v.setInitialLearningRate(args.learningRate)
    n2v.setNetDepth(args.netDepth)
    n2v.setKernelSize(args.netKernelSize)
    n2v.setUNetNFirst(args.unetNFirst)
    n2v.setPercentPixel(args.n2vPercPix)
    if args.noAugment:
        n2v.deactivateDataAugmentation()
    else:
        n2v.activateDataAugmentation()

    n2v.train()
    n2v.saveHistory()
    n2v.printElapsedTime()
    print("---training done---")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main(sys.argv)


