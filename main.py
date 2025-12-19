import os
import collections
import yaml
import numpy as np
import dm_control.suite.swimmer as swimmer
from dm_control.rl import control
from dm_control.utils import rewards
from dm_control import suite
import tonic
import tonic.torch
import torch
from model import SwimmerModule, SwimmerActor

# ==============================================================================
# 1. Environment Setup
# ==============================================================================

_SWIM_SPEED = 0.1

class SwimTargetForward(swimmer.Swimmer):
    """Task to swim towards target with reward for moving forward."""
    def __init__(self, desired_speed=_SWIM_SPEED, **kwargs):
        super().__init__(**kwargs)
        self._desired_speed = desired_speed

    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        physics.model.opt.viscosity = 0.00001
        # Set target position
        xpos, ypos = 1, 1
        physics.named.model.geom_pos['target', 'x'] = xpos
        physics.named.model.geom_pos['target', 'y'] = ypos
        
        # Hide target visuals slightly
        physics.named.model.mat_rgba['target', 'a'] = 1

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['joints'] = physics.joints()
        obs['body_velocities'] = physics.body_velocities()
        return obs

    def get_reward(self, physics):
        # Reward for speed
        forward_velocity = -physics.named.data.sensordata['head_vel'][1]
        forward_reward = rewards.tolerance(
            forward_velocity,
            bounds=(self._desired_speed, float('inf')),
            margin=self._desired_speed,
            value_at_margin=0.,
            sigmoid='linear',
        )
        
        # Reward for target distance
        target_size = physics.named.model.geom_size['target', 0]
        target_reward = rewards.tolerance(
            physics.nose_to_target_dist(),
            bounds=(0, target_size),
            margin=5 * target_size,
            sigmoid='linear'
        )
        
        target_reached = 50 if physics.nose_to_target_dist() < 5 else 0
        return 5*forward_velocity + 30*target_reward + target_reached

@swimmer.SUITE.add()
def swim(n_links=6, desired_speed=_SWIM_SPEED, time_limit=swimmer._DEFAULT_TIME_LIMIT, random=None, environment_kwargs={}):
    """Returns the Swim task for a n-link swimmer."""
    model_string, assets = swimmer.get_model_and_assets(n_links)
    physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
    task = SwimTargetForward(desired_speed=desired_speed, random=random)
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=swimmer._CONTROL_TIMESTEP, **environment_kwargs
    )

# ==============================================================================
# 2. Training Wrapper (FIXED)
# ==============================================================================

def train_agent(header, agent, environment, name='experiment', trainer='tonic.Trainer(steps=int(1e6), save_steps=int(5e4))'):
    """
    Wrapper to train an agent using Tonic RL.
    """
    # 1. Capture configuration immediately for config.yaml
    # We must use 'agent' and 'environment' as argument names so they match what visualize.py expects.
    config = dict(locals())

    # 2. Execute imports
    if header: exec(header)

    # 3. Distribute environment
    env_str = environment
    env_builder = tonic.environments.distribute(lambda: eval(env_str))
    test_env_builder = tonic.environments.distribute(lambda: eval(env_str))

    # 4. Initialize Agent
    agent_str = agent
    agent_instance = eval(agent_str)
    agent_instance.initialize(observation_space=test_env_builder.observation_space, action_space=test_env_builder.action_space)

    # 5. Setup Paths & Logger
    # CRITICAL FIX: passing 'config' here ensures config.yaml is generated
    path = os.path.join('data', 'local', 'experiments', 'tonic', name)
    tonic.logger.initialize(path, script_path=None, config=config)

    # 6. Initialize Trainer
    trainer_instance = eval(trainer)
    trainer_instance.initialize(agent=agent_instance, environment=env_builder, test_environment=test_env_builder)

    # 7. Run
    print(f"Starting training for {name}...")
    trainer_instance.run()
    print("Training complete.")

# ==============================================================================
# 3. Execution
# ==============================================================================

if __name__ == "__main__":
    
    # --------------------------------------------------------------------------
    # 1. Baseline MLP Agent
    # --------------------------------------------------------------------------
    mlp_agent = (
        "tonic.torch.agents.PPO("
        "model=tonic.torch.models.ActorCritic("
        "actor=tonic.torch.models.Actor("
        "encoder=tonic.torch.models.ObservationEncoder(),"
        "torso=tonic.torch.models.MLP(sizes=(64, 64), activation=torch.nn.ReLU),"
        "head=tonic.torch.models.DetachedScaleGaussianPolicyHead()"
        "),"
        "critic=tonic.torch.models.Critic("
        "encoder=tonic.torch.models.ObservationEncoder(),"
        "torso=tonic.torch.models.MLP(sizes=(64, 64), activation=torch.nn.ReLU),"
        "head=tonic.torch.models.ValueHead()"
        ")"
        ")"
        ")"
    )

    # --------------------------------------------------------------------------
    # 2. Biological NCAP Agent
    # --------------------------------------------------------------------------
    # Uses custom SwimmerActor from model.py
    ncap_agent = (
        "tonic.torch.agents.PPO("
        "model=tonic.torch.models.ActorCritic("
        "actor=SwimmerActor(swimmer=SwimmerModule(n_joints = 5)),"
        "critic=tonic.torch.models.Critic("
        "encoder=tonic.torch.models.ObservationEncoder(),"
        "torso=tonic.torch.models.MLP(sizes=(64, 64), activation=torch.nn.ReLU),"
        "head=tonic.torch.models.ValueHead()"
        ")"
        ")"
        ")"
    )

    # --------------------------------------------------------------------------
    # 3. Run Training
    # --------------------------------------------------------------------------
    
    # Train Baseline MLP
    print(">>> Training Baseline MLP...")
    train_agent(
        header='import tonic.torch; import torch', 
        agent=mlp_agent,
        environment='tonic.environments.ControlSuite("swimmer-swimmer6", time_feature=True)',
        name='mlp_baseline'
    )

    # Train Biological NCAP
    print("\n>>> Training Biological NCAP...")
    train_agent(
        header='import tonic.torch; import torch; from model import SwimmerActor',
        agent=ncap_agent,
        environment='tonic.environments.ControlSuite("swimmer-swimmer6", time_feature=True)',
        name='ncap_biological'
    )