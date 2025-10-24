import d3rlpy

class DiscreteOfflineTrainer:
    def __init__(self,algoconfig,fitconfig,modelpath):
        ...
        self.fitconfig = fitconfig
        self.model_path = modelpath
        # could create the config here if convenient
        # the config can be given as an argument or better yet, read from dataclass
        # cql = d3rlpy.algos.DiscreteCQLConfig().create(device = .device)
        # could also read/get the config from somewhere else
        

    def load_model(self):
        ...
        # load the model from an existing file
        """
        algo = d3rlpy.algos.DiscreteCQLConfig().create(device="cpu")
        # Build network shapes from the environment, then load weights
        algo.build_with_env(env)
        algo.load_model(model_path)
        """

    def fit_model(self):
        ...
        # fit the algorithm generally
        # algo.fit(parameters here)

    def save_model(self):
        ...
        #algo.save_model(self.model_path)