from .dl4mic.n2v import N2VNetwork
import warnings

warnings.filterwarnings("ignore")
N2VNetwork.reportGPUAccess()
N2VNetwork.reportDevices()
N2VNetwork.reportTensorFlowAccess()
n2v = N2VNetwork("n2v_synthetic_spots_01")
n2v.setTrainingSource("../../n2v-data/training")
n2v.setPath("../../models")

n2v.setNumberOfEpochs(1)
n2v.setPatchSize(64)
n2v.setBatchSize(128)
n2v.setPercentValidation(10)

n2v.train()
n2v.saveHistory()
n2v.printElapsedTime()
print("trainN2V done")

