import numpy as np
try:
    import gymnasium as gym
except ImportError:
    try:
        import gym
    except ImportError:
        gym = None
try:
    import retro
except ImportError:
    retro = None
import os
import sys
import threading
import gc

# Add external references to path
current_dir = os.path.dirname(os.path.abspath(__file__))
external_ref_path = os.path.join(current_dir, '..', 'external-references')
if external_ref_path not in sys.path:
    sys.path.insert(0, external_ref_path)

try:
    from retro_env import RetroEnv
except ImportError:
    RetroEnv = None

# 全局模拟器管理
_global_emulator_lock = threading.Lock()
_global_current_emulator = None


class StableRetro(gym.Env):
    metadata = {}

    def __init__(
        self,
        game,
        state=None,
        scenario=None,
        info=None,
        use_restricted_actions=None,
        players=1,
        inttype=None,
        obs_type=None,
        action_repeat=1,
        size=(84, 84),
        grayscale=False,
        seed=None,
    ):
        if retro is None:
            raise ImportError(
                "stable-retro is not installed. Please install it with 'pip install stable-retro'"
            )
        if gym is None:
            raise ImportError(
                "gym is not installed. Please install it with 'pip install gym'"
            )
        
        # Set default values
        if state is None:
            state = retro.State.DEFAULT
        if use_restricted_actions is None:
            use_restricted_actions = retro.Actions.FILTERED
        if inttype is None:
            inttype = retro.data.Integrations.STABLE
        if obs_type is None:
            obs_type = retro.Observations.IMAGE
        
        self._game = game
        self._state = state
        self._scenario = scenario
        self._info = info
        self._use_restricted_actions = use_restricted_actions
        self._players = players
        self._inttype = inttype
        self._obs_type = obs_type
        self._action_repeat = action_repeat
        self._size = size
        self._grayscale = grayscale
        self._seed = seed
        
        # Create the environment with global emulator management
        global _global_emulator_lock, _global_current_emulator
        
        with _global_emulator_lock:
            # Close existing global emulator if any
            if _global_current_emulator is not None:
                try:
                    _global_current_emulator.close()
                except:
                    pass
                _global_current_emulator = None
            
            gc.collect()
            
            # Create the environment
            if RetroEnv is not None:
                # Use the more complete RetroEnv implementation
                self._env = RetroEnv(
                    game=game,
                    state=state,
                    scenario=scenario,
                    info=info,
                    use_restricted_actions=use_restricted_actions,
                    players=players,
                    inttype=inttype,
                    obs_type=obs_type,
                    render_mode=None,  # Disable rendering to avoid display issues
                )
            else:
                # Fallback to retro.make
                try:
                    self._env = retro.make(
                        game=game,
                        state=state,
                        scenario=scenario,
                        info=info,
                        use_restricted_actions=use_restricted_actions,
                        players=players,
                        inttype=inttype,
                        obs_type=obs_type,
                        render_mode=None,  # Disable rendering to avoid display issues
                    )
                except TypeError:
                    # Fallback for older versions without render_mode
                    self._env = retro.make(
                        game=game,
                        state=state,
                        scenario=scenario,
                        info=info,
                        use_restricted_actions=use_restricted_actions,
                        players=players,
                        inttype=inttype,
                        obs_type=obs_type,
                    )
            
            # Update global emulator reference
            _global_current_emulator = self._env
        
        # Set up observation and action spaces
        self._setup_spaces()
        
        # Set seed
        if seed is not None:
            self._env.action_space.seed(seed)
            try:
                self._env.seed(seed)
            except AttributeError:
                # For newer gym versions that don't have seed method
                pass
    
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Action space - always convert to Discrete for DreamerV3 compatibility
        if hasattr(self._env.action_space, 'n'):
            # MultiBinary space
            n_buttons = self._env.action_space.n
            # Create discrete space with 2^n_buttons possible combinations
            # But limit to reasonable number to avoid memory issues
            if n_buttons <= 10:  # Max 1024 combinations
                n_actions = 2 ** n_buttons
            else:
                # For too many buttons, use filtered actions if available
                n_actions = min(2 ** n_buttons, 512)  # Cap at 512 actions
            
            self.action_space = gym.spaces.Discrete(n_actions)
            self._n_buttons = n_buttons
            self._is_multibinary = True
        else:
            self.action_space = self._env.action_space
            self._is_multibinary = False
        
        # Observation space - return as dict for DreamerV3 compatibility
        if self._obs_type == retro.Observations.IMAGE:
            # Image observations
            orig_shape = self._env.observation_space.shape
            if self._grayscale:
                # Convert to grayscale
                if len(orig_shape) == 3 and orig_shape[2] == 3:
                    new_shape = tuple(self._size) + (1,)
                else:
                    new_shape = tuple(self._size) + (orig_shape[2],)
            else:
                if len(orig_shape) == 3:
                    new_shape = tuple(self._size) + (orig_shape[2],)
                else:
                    new_shape = tuple(self._size) + (3,)
            
            image_space = gym.spaces.Box(
                low=0, high=255, shape=new_shape, dtype=np.uint8
            )
            self.observation_space = gym.spaces.Dict({
                "image": image_space,
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            })
        else:
            # RAM observations or other types
            self.observation_space = gym.spaces.Dict({
                "image": self._env.observation_space,
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            })
    
    def _convert_action(self, action):
        """Convert action from Discrete to MultiBinary if needed"""
        if self._is_multibinary:
            # Convert discrete action to binary array
            binary_action = np.zeros(self._n_buttons, dtype=np.int8)
            action_int = int(action)
            for i in range(self._n_buttons):
                binary_action[i] = (action_int >> i) & 1
            return binary_action
        else:
            return action
    
    def _process_observation(self, obs):
        """Process observation (resize, grayscale conversion)"""
        if self._obs_type == retro.Observations.IMAGE:
            # Resize observation
            if obs.shape[:2] != self._size:
                try:
                    import cv2
                    obs = cv2.resize(obs, self._size, interpolation=cv2.INTER_AREA)
                except ImportError:
                    from PIL import Image
                    obs_pil = Image.fromarray(obs)
                    obs_pil = obs_pil.resize(self._size, Image.LANCZOS)
                    obs = np.array(obs_pil)
            
            # Convert to grayscale if needed
            if self._grayscale and len(obs.shape) == 3 and obs.shape[2] == 3:
                # Convert RGB to grayscale
                obs = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
                obs = obs.astype(np.uint8)
                obs = np.expand_dims(obs, axis=-1)
        
        return obs
    
    def step(self, action):
        """Execute action for action_repeat steps"""
        # Convert action if needed
        converted_action = self._convert_action(action)
        
        total_reward = 0.0
        raw_obs = None
        for _ in range(self._action_repeat):
            step_result = None
            max_attempts = 3
            attempt = 0
            
            while attempt < max_attempts:
                try:
                    # Check if environment has emulator before stepping
                    if hasattr(self._env, 'em') and self._env.em is not None:
                        step_result = self._env.step(converted_action)
                        break
                    else:
                        print(f"Emulator missing, attempting recovery (attempt {attempt + 1})...")
                        self._reinit_env()
                        attempt += 1
                except AttributeError as e:
                    if "'RetroEnv' object has no attribute 'em'" in str(e) or "em" in str(e):
                        print(f"Emulator error, attempting recovery (attempt {attempt + 1})...")
                        self._reinit_env()
                        attempt += 1
                    else:
                        raise
                except Exception as e:
                    print(f"Unexpected error in retro environment step: {e}")
                    if attempt < max_attempts - 1:
                        self._reinit_env()
                        attempt += 1
                    else:
                        raise
            
            if step_result is None:
                raise RuntimeError("Failed to recover emulator after multiple attempts")
            
            if len(step_result) == 5:
                # New gym API: (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                # Old gym API: (obs, reward, done, info)
                obs, reward, done, info = step_result
            
            total_reward += reward
            # Store the raw observation from the last step for visual reward
            raw_obs = obs.copy() if hasattr(obs, 'copy') else obs
            if done:
                break
        
        # Store raw observation for visual reward computation
        self._last_raw_obs = raw_obs
        
        # Process observation
        obs = self._process_observation(obs)
        
        # Return observation in dict format expected by DreamerV3
        return {
            "image": obs,
            "is_first": False,
            "is_last": done,
            "is_terminal": done,  # For retro games, terminal usually means end of episode
        }, total_reward, done, info
    
    def reset(self, **kwargs):
        """Reset environment"""
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Check if environment has emulator before resetting
                if hasattr(self._env, 'em') and self._env.em is not None:
                    obs = self._env.reset(**kwargs)
                    if isinstance(obs, tuple):
                        obs = obs[0]  # Handle new gym API that returns (obs, info)
                    
                    # Store raw observation for visual reward computation
                    self._last_raw_obs = obs.copy() if hasattr(obs, 'copy') else obs
                    
                    obs = self._process_observation(obs)
                    
                    # Return observation in dict format expected by DreamerV3
                    return {
                        "image": obs,
                        "is_first": True,
                        "is_last": False,
                        "is_terminal": False,
                    }
                else:
                    print(f"Emulator missing during reset, attempting recovery (attempt {attempt + 1})...")
                    self._reinit_env()
                    attempt += 1
            except AttributeError as e:
                if "'RetroEnv' object has no attribute 'em'" in str(e) or "em" in str(e):
                    print(f"Emulator error during reset, attempting recovery (attempt {attempt + 1})...")
                    self._reinit_env()
                    attempt += 1
                else:
                    raise
            except Exception as e:
                print(f"Unexpected error during reset: {e}")
                if attempt < max_attempts - 1:
                    self._reinit_env()
                    attempt += 1
                else:
                    raise
        
        raise RuntimeError("Failed to reset environment after multiple attempts")
    
    def render(self, mode='human'):
        """Render environment"""
        return self._env.render(mode=mode)
    
    def get_raw_frame(self):
        """Get the last raw unprocessed frame for visual reward computation"""
        return getattr(self, '_last_raw_obs', None)
    
    def _reinit_env(self):
        """Reinitialize the retro environment to recover from emulator loss"""
        global _global_emulator_lock, _global_current_emulator
        
        print("Reinitializing retro environment...")
        
        # Store current parameters
        game = self._game
        state = self._state
        scenario = self._scenario
        info = self._info
        use_restricted_actions = self._use_restricted_actions
        players = self._players
        inttype = self._inttype
        obs_type = self._obs_type
        
        with _global_emulator_lock:
            # Force garbage collection to clean up any existing emulator instances
            gc.collect()
            
            # Close current environment and global emulator
            try:
                if hasattr(self._env, 'close'):
                    self._env.close()
            except:
                pass
            
            try:
                if _global_current_emulator is not None:
                    _global_current_emulator.close()
                    _global_current_emulator = None
            except:
                pass
            
            # Clear the environment reference
            self._env = None
            gc.collect()
            
            # Recreate environment
            if RetroEnv is not None:
                self._env = RetroEnv(
                    game=game,
                    state=state,
                    scenario=scenario,
                    info=info,
                    use_restricted_actions=use_restricted_actions,
                    players=players,
                    inttype=inttype,
                    obs_type=obs_type,
                    render_mode=None,  # Disable rendering to avoid display issues
                )
            else:
                # Fallback to retro.make
                try:
                    self._env = retro.make(
                        game=game,
                        state=state,
                        scenario=scenario,
                        info=info,
                        use_restricted_actions=use_restricted_actions,
                        players=players,
                        inttype=inttype,
                        obs_type=obs_type,
                        render_mode=None,  # Disable rendering to avoid display issues
                    )
                except TypeError:
                    # Fallback for older versions without render_mode
                    self._env = retro.make(
                        game=game,
                        state=state,
                        scenario=scenario,
                        info=info,
                        use_restricted_actions=use_restricted_actions,
                        players=players,
                        inttype=inttype,
                        obs_type=obs_type,
                    )
            
            # Update global emulator reference
            _global_current_emulator = self._env
            
            # Verify emulator was created successfully
            if not hasattr(self._env, 'em') or self._env.em is None:
                raise RuntimeError("Failed to create emulator instance")
            
            # Reset environment
            self._env.reset()
            print("Retro environment reinitialized successfully")
    
    def close(self):
        """Close environment"""
        global _global_emulator_lock, _global_current_emulator
        
        with _global_emulator_lock:
            try:
                if hasattr(self._env, 'close'):
                    self._env.close()
                if _global_current_emulator == self._env:
                    _global_current_emulator = None
            except:
                pass
    
    def seed(self, seed=None):
        """Set random seed"""
        try:
            return self._env.seed(seed)
        except AttributeError:
            # For newer gym versions that don't have seed method
            if hasattr(self._env.action_space, 'seed'):
                self._env.action_space.seed(seed)
            if hasattr(self._env.observation_space, 'seed'):
                self._env.observation_space.seed(seed)
            return [seed]
    
    @property
    def unwrapped(self):
        """Get unwrapped environment"""
        return self._env.unwrapped


def make_stable_retro_env(game, **kwargs):
    """Convenience function to create stable-retro environment"""
    return StableRetro(game=game, **kwargs)
