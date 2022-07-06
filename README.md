## GAIL Control
GAIL (Generative Adversarial Imitation Learning) is a reinforcement learning algorithm that is part of the inverse reinforcement learning approach. GAIL works by, initially, recovering an expert's cost function using inverse reinfocement learning, and then using standard reinforcement learning to extract a policy from it. Essentially, this allows us to instruct an agent on how not only to imitate an expert's behavior, but also learn from it in order to deal with unforeseen circumstances (e.g., crashing an autonomous vehicle in a driving simulator).

This repository contains a small and fairly simple research project that demonstrates the GAIL algorithm in the HalfCheetah-v2 environment. Whereas, by using a regular reinforcement learning approach, the agent would need to spend a fair amount of time learning how to exactly move in the environment, this approach allows it to learn this almost immediately and enables the agent to focus more on improving its motor control.

## References
* [Generative Adversarial Imitation Learning](https://arxiv.org/pdf/1606.03476.pdf);
* [D4RL documentation](https://sites.google.com/view/d4rl-anonymous/);
* [Tianshou's GAIL example](https://github.com/thu-ml/tianshou/blob/master/examples/inverse/irl_gail.py).