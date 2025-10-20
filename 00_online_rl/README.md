# README
This is just a messy repo to learn how to do RL, the code quality is rather low, as the objective is to learn as much about RL as possible in minimum time. Once projects become more complex I will take into account coding best practices :)


## Learned lessons

### Hyperparameters
- ent_coef -> leave default, at 0.0, unless the learning curve gets stuck at a local maximum. If ent_coef is too high, it is quite likely that the agent will experience catastrophic forgetting. Important for debugging

- various clipping parameters(clip_range,max_grad_norm) limit the change that the policy can be subject to. Can be useful to avoid catastrophic forgetting. However, if they are too strict they would kill learning

- learning rate: quite obvious. Too low and learning is slow or inexistent, too high and it is unstable

- n_steps: How many steps belong in a batch. Influences training stability. Too high and it will be slow and gentle, or might not even work. Too low and learning will be spiky.

- vf_coef: Relates to the ipmortance of value function vs policy. Have not experimented much with it

### Tensorboard
Fundamental to evaluate how learning works. Easy and intuitive to use. Can add functionality by overriding/adding other classes (see AcroBot.py).

Pay attention to training times, maybe the code changes have made learning inefficient

### Multiple environment training
Not as easy as it may seem at first. Ensure you add a "rank" for the Monitor Wrapper when using SubprocVecEnv, otherwise may be overwriting Monitor.csv file. Generally be careful with the process. Use a single env and it will take ages, use too many envs and you will have way too much overhead. Prefer SubprocVecEnv to use multiple cores. Need to understand inner workings a bit better to implement it myself.

### Callbacks
Experiment with them later today
Seem quite useful
Some of them could even be a default, will try to make my own selection of them

- progress_bar = True (parameter when using model.learn()) -> useful visual of the progress
- checkpoint callback: save the model every certain n of steps
- eval callback: periodically evaluate the agent with a separate test event. Can add other callbacks on top 
- Stop Training on Reward threshold: uses the eval callback (called from it). Stops learning once a reward threshold is achieved
- Every N timesteps: self explanatory
- LogEveryNTimeSteps: same
- StopTrainingonMaxEpisodes: as it sounds. Episodes per env
- StopTrainingOnNoModellImprovement: as it sounds, based on eval callback. needs min n_episodes and n_stale_episodes
		