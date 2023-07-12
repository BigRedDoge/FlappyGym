import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy
#from scipy.stats import loguniform

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),

            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),

            nn.Conv2d(32, 64, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            #1x257664

            #nn.Conv2d(128, 256, kernel_size=3),
            #nn.MaxPool2d(2, 2),
            #nn.ReLU(),
            nn.Flatten(),
        )
       

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        print(x.shape)
        x = self.linear(x)
        print(x.shape)
        return x


class CustomMlp(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomMlp, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.hidden = nn.Sequential(
            nn.Linear(n_input_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),  
            nn.Linear(64, 128),
            nn.ReLU(), 
            nn.Linear(128, 256),
            nn.ReLU(), 
            nn.Flatten()
        )
       

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.hidden(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        #return self.linear(self.cnn(observations))
        #hx, cx = self.lstm(self.linear(self.cnn(observations)))
        return self.linear(self.hidden(observations))


class CustomValuePolicyNetwork(nn.Module):
    
    def __init__(
        self,
        feature_dim: int,
        last_layer_pi: int = 1,
        last_layer_vf: int = 1,
    ):
        super(CustomValuePolicyNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_pi
        self.latent_dim_vf = last_layer_vf

        self.lstm1 = nn.LSTMCell(feature_dim, 1024)
        self.lstm2 = nn.LSTMCell(512, 1024)

        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(64, last_layer_pi),
            #nn.Softmax(dim=1)
            #nn.ReLU()
        )
        """
        self.policy_net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, last_layer_pi)
        )
            #nn.ReLU()
        """
        #)
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(64, last_layer_vf),
            #nn.ReLU()
        )
        """
        self.value_net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU()
        )
            
        """

        self.value_policy_net = nn.Sequential(
            nn.Linear(256, 128),
            #nn.ReLU(),
            nn.LeakyReLU(),

            #nn.Linear(256, 256),
            #nn.ReLU(),
            #nn.LeakyReLU(),

            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.LeakyReLU()
        )
        
        self.input_layer = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LeakyReLU(),

            nn.LSTMCell(512, 256)

            #nn.Linear(256, 512),
            #nn.LeakyReLU(),

            #nn.Linear(512, 1024),
            #nn.LeakyReLU(),
           # 
        )

        
        #)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #self.train(False)
        #features, (hx, cx) = features
        #hx1, cx1 = self.lstm1(features)
        #hx2, cx2 = self.lstm2(hx1)
        x, cx = self.input_layer(features)
        val_pol_net = self.value_policy_net(x)
        policy = self.policy_net(val_pol_net)
        value = self.value_net(val_pol_net)
        return policy, value


class CustomMlpValuePolicyNetwork(nn.Module):
    
    def __init__(
        self,
        feature_dim: int,
        last_layer_pi: int = 32,
        last_layer_vf: int = 32,
    ):
        super(CustomMlpValuePolicyNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_pi
        self.latent_dim_vf = last_layer_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(64, last_layer_pi),
            #nn.Softmax(dim=1)
            #nn.ReLU()
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(64, last_layer_vf),
            #nn.ReLU()
        )


        self.value_policy_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 64),
            nn.LeakyReLU()
        )
        
        self.input_layer = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LeakyReLU(),

            nn.LSTMCell(512, 256)
        )

        
        #)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, cx = self.input_layer(features)
        val_pol_net = self.value_policy_net(x)
        policy = self.policy_net(val_pol_net)
        value = self.value_net(val_pol_net)
        return policy, value

class CustomActorCriticPolicy(ActorCriticCnnPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomValuePolicyNetwork(self.features_dim)



class CustomActorCriticPolicyMlp(ActorCriticCnnPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicyMlp, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomMlpValuePolicyNetwork(self.features_dim)



def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        val = progress_remaining * initial_value
        if val <= final_value:
            val = final_value
        return val

    return func

def log_uni_schedule(a: float, b: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return loguniform(a, b).rvs() * progress_remaining

    return func