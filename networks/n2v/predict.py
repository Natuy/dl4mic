from dl4mic.n2v import N2VNetwork

n2v = N2VNetwork("n2v_synthetic_spots_01")
n2v.setPath("../../models")
n2v.predict("../../n2v-data/prediction-in", "../../n2v-data/prediction-out")
