"""Base environment class for OpenEnv"""


class BaseEnvironment:
    """Base class for OpenEnv environments"""
    
    def reset(self, **kwargs):
        """Reset environment to initial state
        
        Returns:
            Initial observation/state
        """
        raise NotImplementedError
    
    def step(self, action):
        """Execute one step in the environment
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        raise NotImplementedError
    
    def get_state(self):
        """Get current state without modifying environment
        
        Returns:
            Current state
        """
        raise NotImplementedError
