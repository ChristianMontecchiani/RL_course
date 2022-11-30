import dataclasses
import copy
import gym

## Simulator
@dataclasses.dataclass
class Checkpoint:
    """Holds the checkpoint state for the environment simulator."""
    needs_reset: bool
    env: gym.Env

class SimulatorWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        # assert isinstance(env, gym.Env), "env should be gym.Env"
        self._env = copy.deepcopy(env)
        self._needs_reset = True

        self.save_checkpoint()

    def __getattr__(self, attr: str):
        # Delegates attribute calls to the wrapped environment.
        return getattr(self._env, attr)

    # Getting/setting of state is necessary so that getattr doesn't delegate them
    # to the wrapped environment. This makes sure pickling a wrapped environment
    # works as expected.
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save_checkpoint(self,):
        self._checkpoint = Checkpoint(
            needs_reset = self._needs_reset,
            env = copy.deepcopy(self._env)
        )
    
        return self._checkpoint
    
    def load_checkpoint(self,):
        self._env = copy.deepcopy(self._checkpoint.env)
        self._needs_reset = self._checkpoint.needs_reset

    def step(self, action):
        if self._needs_reset:
            raise ValueError('This model needs to be explicitly reset.')
        obs, r, d, info = self._env.step(action)

        self._needs_reset = d
        return obs, r, d, info

    def reset(self):
        self._needs_reset = False
        return self._env.reset()

    @property
    def needs_reset(self):
        return self._needs_reset

