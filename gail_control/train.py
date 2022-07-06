import os
import gym
import d4rl
import torch
import pprint
import numpy as np
import tianshou as ts
from torch import nn
from env_wrapper import make_env
from tianshou.policy import GAILPolicy
from torch.optim.lr_scheduler import LambdaLR
from tianshou.trainer import onpolicy_trainer
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Independent, Normal
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.data import Batch, Collector, ReplayBuffer, VectorReplayBuffer

# Using CUDA if it's available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining some policy hyperparameters.
LR = 3e-4
DISC_LR = 2.5e-5
GAMMA = 0.99
GAE_LAMBDA = 0.95
MAX_GRAD_NORM = 0.5
VF_COEF = 0.25
ENT_COEF = 0.001
DISC_UPDATE_NUM = 2
REW_NORM = True
BOUND_ACTION_METHOD = "clip"

# Defining the training config.
BATCH_SIZE = 64
BUFFER_SIZE = 100000
STEP_PER_EPOCH = 100000
STEP_PER_COLLECT = 1000
REPEAT_PER_COLLECT = 4
TRAIN_NUM = 10
TEST_NUM = int(os.cpu_count()) - 1


def dist(*logits):
    """
    Returning an independent normal distribution based on the logits.
    """
    return Independent(Normal(*logits), 1)


def train(env_name, n_epochs):
    """
    The training method, which expects an env name as input and
    the number of epochs.
    """
    # Creating the model path.
    expert_task = f"{str.lower(env_name).split('-')[0]}-expert"
    save_path = f"gail_control/log/{env_name}_gail"

    try:
        os.mkdir(save_path)
    
    except FileExistsError:
        pass

    env, train_envs, test_envs = make_env(env_name, TRAIN_NUM, TEST_NUM)

    # Getting some info about the environment.
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    # Seeding the experiment and environments.
    np.random.seed(42)
    torch.manual_seed(42)
    train_envs.seed(42)
    test_envs.seed(42)

    # Creating the model, actor and critic networks.
    net = Net(
        state_shape, 
        hidden_sizes=[64, 64],
        activation=nn.Tanh,
        device=device
    ).to(device)
    
    actor = ActorProb(
        net, 
        action_shape,
        max_action=max_action,
        unbounded=True,
        device=device
    ).to(device)

    critic = Critic(net, device=device).to(device)

    # Filling the actor's sigma parameter with constant values (-0.5).
    torch.nn.init.constant_(actor.sigma_param, -0.5)

    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            # Orthogonal initialization.
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    # Taken from Tianshou's IRL_GAIL tutorial:
    # Doing last layer policy scaling. The idea is that it will make the
    # initial actions have around 0 mean and std, which helps in boosting
    # performance. See https://arxiv.org/abs/2006.05990 (fig. 24) for details.
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=LR)

    # Creating the discriminator network.
    net_disc = Net(
        state_shape, 
        action_shape,
        hidden_sizes=[64, 64],
        activation=nn.Tanh,
        device=device,
        concat=True
    ).to(device)

    disc_net = Critic(net_disc, device=device).to(device)

    for m in disc_net.modules():
        if isinstance(m, torch.nn.Linear):
            # Orthogonal initialization for the discriminator.
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    
    disc_optim = torch.optim.Adam(disc_net.parameters(), lr=DISC_LR)

    # Decaying the learning rate linearly to 0.
    max_update_num = np.ceil(STEP_PER_EPOCH / STEP_PER_COLLECT) * n_epochs

    lr_scheduler = LambdaLR(
        optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
    )

    # Creating the expert replay buffer.
    dataset = d4rl.qlearning_dataset(gym.make(expert_task))
    dataset_size = dataset["rewards"].size

    expert_buffer = ReplayBuffer(dataset_size)

    # Adding the expert data to the buffer.
    for i in range(dataset_size):
        expert_buffer.add(
            Batch(
                obs=dataset["observations"][i],
                act=dataset["actions"][i],
                rew=dataset["rewards"][i],
                done=dataset["terminals"][i],
                obs_next=dataset["next_observations"][i]
            )
        )

    print("Dataset loaded.")

    # Creating the policy.
    policy = GAILPolicy(
        actor,
        critic,
        optim,
        dist,
        expert_buffer,
        disc_net,
        disc_optim,
        disc_update_num=DISC_UPDATE_NUM,
        discount_factor=GAMMA,
        gae_lambda=GAE_LAMBDA,
        max_grad_norm=MAX_GRAD_NORM,
        vf_coef=VF_COEF,
        ent_coef=ENT_COEF,
        reward_normalization=REW_NORM,
        action_scaling=True,
        action_bound_method=BOUND_ACTION_METHOD,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space    
    )

    # Verifying if we can resume training.
    try:
        policy.load_state_dict(torch.load(os.path.join(save_path, "policy.pth"), map_location=device))

        print(f"Loaded model from {os.path.join(save_path, 'policy.pth')}, resuming training...")

    except FileNotFoundError:
        pass

    # Creating a buffer and train and test collectors.
    buffer = VectorReplayBuffer(BUFFER_SIZE, len(train_envs))
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)


    def save_best_fn(policy):
        """
        This function saves the best model.
        """
        torch.save(policy.state_dict(), os.path.join(save_path, "policy.pth"))


    # Training the policy.
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        n_epochs,
        STEP_PER_EPOCH,
        REPEAT_PER_COLLECT,
        TEST_NUM,
        BATCH_SIZE,
        STEP_PER_COLLECT,
        save_best_fn=save_best_fn,
        logger=LOGGER,
        test_in_train=False
    )

    pprint.pprint(result)


if __name__ == "__main__":
    name = "HalfCheetah-v2"

    LOGGER = ts.utils.TensorboardLogger(SummaryWriter(f"gail_control/log/{name}_gail"))

    # Training the agent.
    train(name, n_epochs=20)
