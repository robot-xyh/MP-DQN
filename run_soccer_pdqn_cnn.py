import os
import click
import time
from common import ClickPythonLiteralOption
from common.soccer_domain import SoccerScaledParameterisedActionWrapper
import gym
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from gym.wrappers import Monitor
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import airgym
writer = SummaryWriter()
"""
def pad_action(act, act_param):
    action = np.zeros((8,))
    action[0] = act
    if act == 0:
        action[[1,2]] = act_param
    elif act == 1:
        action[3] = act_param
    elif act == 2:
        action[[4,5]] = act_param
    elif act == 3:
        action[[6]] = act_param
    else:
        raise ValueError("Unknown action index '{}'".format(act))
    return action
"""
def pad_action(act, act_param):
    params = [np.zeros((3,)),np.zeros((3,)),np.zeros((3,)),np.zeros((3,)),np.zeros((3,)),np.zeros((3,)), np.zeros((3,)), np.zeros((3,))]
    params[act] = act_param
    return (act, params)

def evaluate(env, agent, episodes=10):
    returns = []
    timesteps = []
    goals = []
    for _ in range(episodes):
        state = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        info = {'status': "NOT_SET"}
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(state)
            action = pad_action(act, act_param)
            state, reward, terminal, info = env.step(action)
            total_reward += reward
        # print(info['status'])
        goal = info['status'] == 'GOAL'
        timesteps.append(t)
        returns.append(total_reward)
        goals.append(goal)
    return np.column_stack((returns, timesteps, goals))


def make_env(scale_actions):
    # env = gym.make('SoccerEmptyGoal-v0')
    env = gym.make("airgym:airsim-drone-sample-v0",
                ip_address="127.0.0.1",
                step_length=0.25,
                image_shape=(1,84,84),)


@click.command()
@click.option('--seed', default=0, help='Random seed.', type=int)
@click.option('--episodes', default=20000, help='Number of epsiodes.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--batch-size', default=128, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.99, help='Discount factor.', type=float)
@click.option('--update-ratio', default=0.1, help='Ratio of updates to samples.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=1000, help='Number of transitions required to start learning.',
              type=int)
@click.option('--use-ornstein-noise', default=False,
              help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
@click.option('--replay-memory-size', default=10000, help='Replay memory size in transitions.', type=int) # 500000
@click.option('--epsilon-steps', default=10000, help='Number of episodes over which to linearly anneal epsilon.', type=int)
@click.option('--epsilon-final', default=0.01, help='Final epsilon value.', type=float)
@click.option('--tau-actor', default=0.001, help='Soft target network update averaging factor.', type=float)
@click.option('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.', type=float)  # 0.001
@click.option('--learning-rate-actor', default=0.001, help="Actor network learning rate.", type=float)
@click.option('--learning-rate-actor-param', default=0.00001, help="Critic network learning rate.", type=float)
@click.option('--clip-grad', default=10., help="Gradient clipping.", type=float)  # 1 better than 10.
@click.option('--beta', default=0.2, help='Averaging factor for on-policy and off-policy targets.', type=float)  # 0.5
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--split', default=False, help='Separate action-parameter inputs.', type=bool)
@click.option('--multipass', default=False, help='Separate action-parameter inputs using multiple Q-network passes.', type=bool)
@click.option('--indexed', default=False, help='Indexed loss function.', type=bool)
@click.option('--weighted', default=False, help='Naive weighted loss function.', type=bool)
@click.option('--average', default=False, help='Average weighted loss function.', type=bool)
@click.option('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
@click.option('--zero-index-gradients', default=False, help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.", type=bool)
@click.option('--action-input-layer', default=0, help='Which layer to input action parameters at when using split Q-networks.', type=int)
@click.option('--layers', default="[256,128,64]", help='Duplicate action-parameter inputs.', cls=ClickPythonLiteralOption)
@click.option('--save-freq', default=0, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="results/soccer", help='Output directory.', type=str)
@click.option('--title', default="PDQN", help="Prefix of output files", type=str)
def run(seed, episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold, replay_memory_size,
        epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor, learning_rate_actor_param, title, epsilon_final,
        clip_grad, beta, scale_actions, split, indexed, zero_index_gradients, action_input_layer,
        evaluation_episodes, multipass, weighted, average, random_weighted, update_ratio,
        save_freq, save_dir, layers):

    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
        os.makedirs(save_dir, exist_ok=True)

    env = make_env(scale_actions)
    dir = os.path.join(save_dir, title)
    env = gym.make("airgym:airsim-drone-sample-v0",
                ip_address="127.0.0.1",
                step_length=0.25,
                image_shape=(7,84,84),)
    env = Monitor(env, directory=os.path.join(dir, str(seed)), video_callable=False, write_upon_reset=False, force=True)
    # env.seed(seed)  # doesn't work on HFO
    np.random.seed(seed)

    from agents.pdqn_CNN import PDQNAgent
    from agents.pdqn_split_nstep import PDQNNStepSplitAgent
    from agents.pdqn_multipass_nstep import MultiPassPDQNNStepAgent
    assert not (split and multipass)
    agent_class = PDQNAgent
    if split:
        agent_class = PDQNNStepSplitAgent
    elif multipass:
        agent_class = MultiPassPDQNNStepAgent
    assert action_input_layer >= 0
    if action_input_layer > 0:
        assert split
    agent = agent_class(
                           env.observation_space, env.action_space,
                           actor_kwargs={"hidden_layers": layers,
                                         'action_input_layer': action_input_layer,
                                         'activation': "leaky_relu",
                                         'output_layer_init_std': 0.01},
                           actor_param_kwargs={"hidden_layers": layers,
                                               'activation': "leaky_relu",
                                               'output_layer_init_std': 0.01},
                           batch_size=batch_size,
                           learning_rate_actor=learning_rate_actor,  # 0.0001
                           learning_rate_actor_param=learning_rate_actor_param,  # 0.001
                           epsilon_steps=epsilon_steps,
                           epsilon_final=epsilon_final,
                           gamma=gamma,  # 0.99
                           tau_actor=tau_actor,
                           tau_actor_param=tau_actor_param,
                           clip_grad=clip_grad,
                           
                           indexed=indexed,
                           weighted=weighted,
                           average=average,
                           random_weighted=random_weighted,
                           initial_memory_threshold=initial_memory_threshold,
                           use_ornstein_noise=use_ornstein_noise,
                           replay_memory_size=replay_memory_size,
                           inverting_gradients=inverting_gradients,
                           zero_index_gradients=zero_index_gradients,
                           seed=seed)
    print(agent)
    network_trainable_parameters = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
    network_trainable_parameters += sum(p.numel() for p in agent.actor_param.parameters() if p.requires_grad)
    print("Total Trainable Network Parameters: %d" % network_trainable_parameters)
    max_steps = 15000
    total_reward = 0.
    returns = []
    timesteps = []
    goals = []
    start_time_train = time.time()
    
    infogs =[]
    infoms =[]
    infows = []
    
    meaninfogs =[]
    meaninfoms =[]
    meaninfows = []
    k = 0 
    for i in range(episodes):
        
        infogs.append([])
        infoms.append([])
        infows.append([])
        if save_freq > 0 and save_dir and i % save_freq == 0:
            agent.save_models(os.path.join(save_dir, str(i)))
        state = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        #if visualise and i % render_freq == 0:
        #    env.render()

        act, act_param, all_action_parameters = agent.act(state)
        action = pad_action(act, act_param)
        
        episode_reward = 0.
        agent.start_episode()
        for j in range(max_steps):

            ret = env.step(action)
            k +=1
            next_state, reward, terminal, info = ret
            infoms[i].append(info["dist"])
            infogs[i].append(info["angle"])
            infows[i].append(info["win"])
            
            next_state = np.array(next_state, dtype=np.float32, copy=False)

            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            next_action = pad_action(next_act, next_act_param)
            agent.step(state, (act, all_action_parameters), reward, next_state,
                       (next_act, next_all_action_parameters), terminal)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state

            episode_reward += reward
            

            if terminal:
                break
        agent.end_episode()
        
        meaninfogs.append(  min(infogs[i]))
        meaninfoms.append(  min(infoms[i]))
        meaninfows.append(  max(infows[i]))
        
        
        returns.append(episode_reward)
        total_reward += episode_reward
        if i % 100 == 0:
            writer.add_scalar('reward', episode_reward,k)   
            writer.add_scalar('length', env.get_episode_lengths()[i],k)
            
            #writer.add_scalar('info', env.get_episode_infos()[i]["win"],i)
            writer.add_scalar('infog', np.array(meaninfogs[-100:]).mean(),k)
            writer.add_scalar('infom', np.array(meaninfoms[-100:]).mean(),k)
            writer.add_scalar('infow', np.array(meaninfows[-100:]).mean(),k)
            #writer.add_scalar('infowmean', np.array(infows[i]).mean(),i)
            print('{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i), total_reward / (i + 1), np.array(returns[-100:]).mean()))
    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()
    if save_freq > 0 and save_dir:
        agent.save_models(os.path.join(save_dir, str(i)))

    returns = env.get_episode_rewards()
    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)

    np.save(os.path.join(dir, title + "{}".format(str(seed))),returns)
    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        agent.epsilon_final = 0.
        agent.epsilon = 0.
        agent.noise = None
        agent.actor.eval()
        agent.actor_param.eval()
        start_time_eval = time.time()
        evaluation_results = evaluate(env, agent, evaluation_episodes)  # returns, timesteps, goals
        end_time_eval = time.time()
        print("Ave. evaluation return =", sum(evaluation_results[:, 0]) / evaluation_results.shape[0])
        print("Ave. timesteps =", sum(evaluation_results[:, 1]) / evaluation_results.shape[0])
        goal_timesteps = evaluation_results[:, 1][evaluation_results[:, 2] == 1]
        if len(goal_timesteps) > 0:
            print("Ave. timesteps per goal =", sum(goal_timesteps) / evaluation_results.shape[0])
        else:
            print("Ave. timesteps per goal =", sum(goal_timesteps) / evaluation_results.shape[0])
        print("Ave. goal prob. =", sum(evaluation_results[:, 2]) / evaluation_results.shape[0])
        np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_results)
        print("Evaluation time: %.2f seconds" % (end_time_eval - start_time_eval))
    print("Training time: %.2f seconds" % (end_time_train - start_time_train))

    print(agent)
    env.close()


def compute_n_step_returns(episode_transitions, gamma):
    n = len(episode_transitions)
    n_step_returns = np.zeros((n,))
    n_step_returns[n-1] = episode_transitions[n-1][2]  # Q-value is just the final reward
    for i in range(n-2, 0, -1):
        reward = episode_transitions[i][2]
        target = n_step_returns[i + 1]
        n_step_returns[i] = reward + gamma * target
    return n_step_returns


if __name__ == '__main__':
    run()
