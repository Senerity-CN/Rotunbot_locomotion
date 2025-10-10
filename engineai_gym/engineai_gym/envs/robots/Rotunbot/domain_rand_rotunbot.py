import torch
from engineai_gym.envs.base.domain_rands.domain_rands_base import DomainRandsBase


class SphericalRobotDomainRand(DomainRandsBase):
    """Domain randomization class for the spherical robot."""

    def __init__(self, env):
        """Initialize domain randomization handler.

        Args:
            env: Environment instance
        """
        super().__init__(env)
        self.env = env
        self.domain_rand_vec = {}

    def add_domain_rand(self, name, range, distribution="uniform", delta_range=None):
        """Add a domain randomization parameter.
        
        Args:
            name: Name of the parameter
            range: Range of values [min, max]
            distribution: Distribution type ("uniform" or "normal")
            delta_range: Range for delta updates [min, max]
        """
        self.domain_rand_vec[name] = {
            "range": range,
            "distribution": distribution,
            "delta_range": delta_range
        }

    def init_domain_rands(self):
        """Initialize domain randomization parameters."""
        # Initialize randomization ranges based on config
        if getattr(self.env.cfg.domain_rand, 'randomize_friction', True):
            self.add_domain_rand("friction",
                               range=[0.8, 1.2],
                               distribution="uniform",
                               delta_range=[-0.05, 0.05])

        if getattr(self.env.cfg.domain_rand, 'randomize_restitution', True):
            self.add_domain_rand("restitution",
                               range=[0.0, 0.2],
                               distribution="uniform",
                               delta_range=[-0.02, 0.02])

        if getattr(self.env.cfg.domain_rand, 'randomize_base_mass', False):
            self.add_domain_rand("base_mass",
                               range=self.env.cfg.domain_rand.added_mass_range,
                               distribution="uniform",
                               delta_range=[-0.02, 0.02])

        if getattr(self.env.cfg.domain_rand, 'randomize_link_mass', False):
            self.add_domain_rand("link_mass",
                               range=self.env.cfg.domain_rand.added_link_mass_range,
                               distribution="uniform",
                               delta_range=[-0.02, 0.02])

        if getattr(self.env.cfg.domain_rand, 'randomize_damping', True):
            self.add_domain_rand("damping",
                               range=[0.9, 1.1],
                               distribution="uniform",
                               delta_range=[-0.05, 0.05])

    def process_on_reset_idx(self, env_ids):
        """Apply domain randomization on environment reset.

        Args:
            env_ids: Environment IDs to apply randomization to
        """
        if len(env_ids) == 0:
            return

        # For now, we'll just log that domain randomization is being applied
        # In a full implementation, this would modify the environment properties
        print(f"Applying domain randomization to environments: {env_ids}")

    def process_on_step(self):
        """Apply domain randomization on each step (if needed)."""
        # For spherical robot, we typically only randomize on reset
        pass

    def get_domain_rand_names(self):
        """Get names of domain randomization parameters.

        Returns:
            list: List of domain randomization parameter names
        """
        names = []
        if getattr(self.env.cfg.domain_rand, 'randomize_friction', True):
            names.append("friction")
        if getattr(self.env.cfg.domain_rand, 'randomize_restitution', True):
            names.append("restitution")
        if getattr(self.env.cfg.domain_rand, 'randomize_base_mass', False):
            names.append("base_mass")
        if getattr(self.env.cfg.domain_rand, 'randomize_link_mass', False):
            names.append("link_mass")
        if getattr(self.env.cfg.domain_rand, 'randomize_damping', True):
            names.append("damping")
        return names