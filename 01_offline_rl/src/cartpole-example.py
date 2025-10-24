import d3rlpy

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
)

cql.save_model("cql_cartpole.pt")