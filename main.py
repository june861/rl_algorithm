#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2024/09/28 10:10:25
@Author  :   junewluo 
@Email   :   overtheriver861@gmail.com
@description   :   use subprocess lib to build a script.
'''

import subprocess
import multiprocessing
import argparse
import configparser
import os

parser = argparse.ArgumentParser("Start the script")
parser.add_argument("--conf", type=str, default="configure.conf", help="path of configure file")
parser.add_argument("--algorithms", type=str, nargs='+', default=["dqn"], help="which algorithms to use")

def build_dqn_script(conf):

    # usage: DQN Parameter Setting [-h] [--env_name ENV_NAME] [--env_num ENV_NUM] [--max_eposide_step MAX_EPOSIDE_STEP] [--seed SEED] 
    #                               [--max_train_steps MAX_TRAIN_STEPS] [--learn_freq LEARN_FREQ]
    #                             [--evaluate_freq EVALUATE_FREQ] [--evaluate_times EVALUATE_TIMES] [--lr LR] [--gamma GAMMA] 
    #                               [--epsilon EPSILON] [--epsilon_min EPSILON_MIN] [--epsilon_decay EPSILON_DECAY]    
    #                             [--batch_size BATCH_SIZE] [--mini_batch_size MINI_BATCH_SIZE] [--capacity CAPACITY] 
    #                               [--use_lr_decay USE_LR_DECAY] [--update_target UPDATE_TARGET] [--layers LAYERS]
    #                             [--hidden_dims HIDDEN_DIMS [HIDDEN_DIMS ...]] [--wandb WANDB] [--tensorboard TENSORBOARD]

    dqn_main_script = os.path.join(os.getcwd(), "dqn_main.py")
    if not os.path.exists(dqn_main_script):
        raise FileExistsError(f'No such script {dqn_main_script}')
    env_names = conf.get('ENV','env_name').split(',')
    hidden_dims = conf.get('NETWORK', 'hidden_dims').split(",")
    dqn_start_comms = []
    for env_name in env_names:  
        comm = [  
            'python', 'dqn_main.py',  
            '--env_name', env_name,  
            '--env_num', conf.get('ENV', 'env_num'),  
            '--max_eposide_step', conf.get('DQN', 'max_eposide_step'),  
            '--seed', conf.get('DQN', 'seed', fallback='1'),  # 假设有默认值 'None'  
            '--max_train_steps', conf.get('DQN', 'max_train_steps'),  
            '--learn_freq', conf.get('DQN', 'learn_freq'),  
            '--evaluate_freq', conf.get('DQN', 'evaluate_freq'),  
            '--evaluate_times', conf.get('DQN', 'evaluate_times', fallback='1'),  # 假设默认评估次数为1  
            '--lr', conf.get('DQN', 'dqn_lr'),  
            '--gamma', conf.get('DQN', 'gamma'),  
            '--epsilon', conf.get('DQN', 'epsilon'),  
            '--epsilon_min', conf.get('DQN', 'epsilon_min'),  
            '--epsilon_decay', conf.get('DQN', 'epsilon_decay'),  
            '--batch_size', conf.get('DQN', 'batch_size'),  
            '--mini_batch_size', conf.get('DQN', 'mini_batch_size'),  
            '--capacity', conf.get('ENV', 'capacity'),  
            '--use_lr_decay', str(conf.getboolean('DQN', 'use_lr_decay', fallback=False)),  # 假设默认为False  
            '--update_target', conf.get('DQN', 'update_target'),  
            '--layers', conf.get('NETWORK', 'layers'),  
            '--wandb', str(conf.getboolean('MONITOR', 'wandb', fallback=False)),  # 假设默认为False  
            '--tensorboard', str(conf.getboolean('MONITOR', 'tensorboard', fallback=True))  # 假设默认为False  
        ]
        # 添加隐层维度信息
        comm.append('--hidden_dims')
        for dim in hidden_dims:
            comm.append(dim)
        dqn_start_comms.append(comm)

    return dqn_start_comms

def build_ppo_script(conf):
    # usage: Hyperparameter Setting for PPO [-h] 
    # [--env_name ENV_NAME] [--env_num ENV_NUM] [--use_multiprocess USE_MULTIPROCESS] [--max_train_steps MAX_TRAIN_STEPS]                                  
    # [--per_batch_steps PER_BATCH_STEPS] [--evaluate_freq EVALUATE_FREQ] [--save_freq SAVE_FREQ] [--batch_size BATCH_SIZE]
    # [--mini_batch_size MINI_BATCH_SIZE] [--hidden_width HIDDEN_WIDTH] [--lr_a LR_A] [--lr_c LR_C] [--gamma GAMMA] [--lamda LAMDA] [--epsilon EPSILON]  
    # [--use_off_policy USE_OFF_POLICY] [--use_buffer USE_BUFFER] [--use_gae USE_GAE] [--grad_clip_param GRAD_CLIP_PARAM] [--use_adv_norm USE_ADV_NORM]  
    # [--use_state_norm USE_STATE_NORM] [--use_reward_norm USE_REWARD_NORM] [--use_reward_scaling USE_REWARD_SCALING] [--entropy_coef ENTROPY_COEF]      
    # [--use_lr_decay USE_LR_DECAY] [--use_grad_clip USE_GRAD_CLIP] [--use_orthogonal_init USE_ORTHOGONAL_INIT] [--set_adam_eps SET_ADAM_EPS]
    # [--use_tanh USE_TANH] [--use_ppo_clip USE_PPO_CLIP] [--monitor {tensorboard,wandb}]
    ppo_main_script = os.path.join(os.getcwd(), "dqn_main.py")
    if not os.path.exists(ppo_main_script):
        raise FileExistsError(f'No such script {ppo_main_script}')
    env_names = conf.get('ENV','env_name').split(',')
    ppo_start_comms = []
    ppo_conf = conf['PPO']
    for env_name in env_names:
        # add env info
        comm = [
            'python', 'dqn_main.py',  
            '--env_name', env_name, '--env_num', conf.get('ENV', 'env_num'),
        ]
        # add hidden dims
        for key, value in ppo_conf.items():
            comm.append(key)
            comm.append(value)
        ppo_start_comms.append(comm)
    
    return ppo_start_comms

def build_pg_script(conf):
    pass

def build_ddpg_script(conf):
    pass

def run_script(comm, log):
    with open(log, 'w') as f_combined:  
        subprocess.Popen(  
            comm,  
            stdout=f_combined,  
            stderr=subprocess.STDOUT,  # 将stderr重定向到stdout的文件对象  
            shell=False  
        )

def main(args):
    conf_path = args.conf
    current_pwd = os.getcwd()
    if not os.path.exists(os.path.join(current_pwd, conf_path)):
        raise FileExistsError(f'No such file {conf_path}')
    algorithms = args.algorithms
    config = configparser.ConfigParser()
    config.read(conf_path)

    # build executable command
    algorithms_rank = ['pg', 'ppo', 'dqn', 'ddpg']
    build_func = [build_pg_script, build_ppo_script, build_dqn_script, build_ddpg_script]
    run_commands = []
    for algorithm in algorithms:
        index = algorithms_rank.index(algorithm)
        if index == -1:
            continue
        func = build_func[index]
        run_command = func(conf = config)
        run_commands.extend(run_command)
    
    processes = []

    for script in run_commands:
        print(f'scrip of {script[1]} will be run as:\n {script}')
        log = f'logs/{script[1]}_{script[3]}.log'
        p = multiprocessing.Process(target = run_script, args=(script,log))
        processes.append(p)
        p.start() 

if __name__ == '__main__':
    args = parser.parse_args()
    main(args = args)
    