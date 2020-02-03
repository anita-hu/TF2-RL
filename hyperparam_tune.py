import sys
import numpy as np
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'AE-DDPG')

# install hyperopt https://github.com/hyperopt/hyperopt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from TF2_AE_DDPG import AE_DDPG


# hyper parameter space
space = {
    'actor_lr': hp.loguniform('a_lr', np.log(1e-6), np.log(1e-2)),
    'critic_lr': hp.loguniform('c_lr', np.log(1e-6), np.log(1e-2)),
    'sigma_decay': 1 - hp.loguniform('s_decay', np.log(1e-5), np.log(1e-2)),
    # 'batch_size': hp.choice('bs', [32, 64, 128]),
    # 'actor_dense_1': hp.choice('a_L1', range(1, 48)),
    # 'actor_dense_2': hp.choice('a_L2', range(1, 36)),
    # 'critic_dense_1': hp.choice('c_L1', range(1, 48)),
    # 'critic_dense_2': hp.choice('c_L2', range(1, 36)),
}


def f(params, test_trials=5):
    name = "CartPole-v1"
    print(params)
    ddpg = AE_DDPG(name,
                   discrete=True,
                   lr_actor=params['actor_lr'],
                   lr_critic=params['critic_lr'],
                   sigma_decay=params['sigma_decay'],
                   # actor_units=(params['actor_dense_1'], params['actor_dense_2']),
                   # critic_units=(params['critic_dense_1'], params['critic_dense_2'])
                   )
    ddpg.train(max_epochs=1600, save_freq=1000)
    total_loss = 0
    for i in range(test_trials):
        rewards = ddpg.test(render=False)
        total_loss += 500 - rewards  # 500 is max reward

    return {'loss': total_loss/test_trials, 'status': STATUS_OK}


if __name__ == "__main__":
    trials = Trials()
    best = fmin(f, space, algo=tpe.suggest, max_evals=10, trials=trials)
    print("Best: ", space_eval(space, best))
