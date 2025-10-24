import d3rlpy
# Loading a dataset, which is probably perfect
dataset,env = d3rlpy.datasets.get_cartpole()

#setup CQL algorithm
cql = d3rlpy.algos.DiscreteCQLConfig().create(device='cuda:0')

#start training
cql.fit(
    dataset,
    n_steps=10000,
    n_steps_per_epoch=1000,
    evaluators={
        'environment': d3rlpy.metrics.EnvironmentEvaluator(env), # evaluate with CartPole-v1 environment
    },
    logger_adapter=d3rlpy.logging.TensorboardAdapterFactory(root_dir='01_offline_rl/logs'),
)

cql.save_model("01_offline_rl/models/cql_cartpole.pt")