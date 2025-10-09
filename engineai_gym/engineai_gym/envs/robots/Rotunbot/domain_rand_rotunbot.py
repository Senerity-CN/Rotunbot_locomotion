import torch
from engineai_gym.envs.base.domain_rands.domain_rands import DomainRands


class SphericalRobotDomainRand(DomainRands):
    """Domain randomization class for the spherical robot."""

    def __init__(self, env):
        """Initialize domain randomization handler.

        Args:
            env: Environment instance
        """
        super().__init__(env)
        self.env = env

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

        # Apply friction randomization
        if "friction" in self.domain_rand_vec:
            friction_coeffs = self.domain_rand_vec["friction"][env_ids]
            for i, env_id in enumerate(env_ids):
                # Apply to all rigid bodies in the environment
                for j in range(self.env.num_bodies):
                    rb_props = self.env.gym.get_actor_rigid_body_properties(self.env.envs[env_id], 0)
                    for rb_prop in rb_props:
                        rb_prop.friction = friction_coeffs[i]
                    self.env.gym.set_actor_rigid_body_properties(self.env.envs[env_id], 0, rb_props)

        # Apply restitution randomization
        if "restitution" in self.domain_rand_vec:
            restitution_coeffs = self.domain_rand_vec["restitution"][env_ids]
            for i, env_id in enumerate(env_ids):
                # Apply to all rigid bodies in the environment
                for j in range(self.env.num_bodies):
                    rb_props = self.env.gym.get_actor_rigid_body_properties(self.env.envs[env_id], 0)
                    for rb_prop in rb_props:
                        rb_prop.restitution = restitution_coeffs[i]
                    self.env.gym.set_actor_rigid_body_properties(self.env.envs[env_id], 0, rb_props)

        # Apply base mass randomization
        if "base_mass" in self.domain_rand_vec and getattr(self.env.cfg.domain_rand, 'randomize_base_mass', False):
            mass_additions = self.domain_rand_vec["base_mass"][env_ids]
            for i, env_id in enumerate(env_ids):
                # Apply to base link (index 0)
                rb_props = self.env.gym.get_actor_rigid_body_properties(self.env.envs[env_id], 0)
                for rb_prop in rb_props:
                    rb_prop.mass += mass_additions[i]
                self.env.gym.set_actor_rigid_body_properties(self.env.envs[env_id], 0, rb_props)

        # Apply link mass randomization
        if "link_mass" in self.domain_rand_vec and getattr(self.env.cfg.domain_rand, 'randomize_link_mass', False):
            mass_multipliers = self.domain_rand_vec["link_mass"][env_ids]
            for i, env_id in enumerate(env_ids):
                # Apply to all links except base (index > 0)
                rb_props = self.env.gym.get_actor_rigid_body_properties(self.env.envs[env_id], 0)
                for j, rb_prop in enumerate(rb_props):
                    if j > 0:  # Not the base link
                        rb_prop.mass *= mass_multipliers[i]
                self.env.gym.set_actor_rigid_body_properties(self.env.envs[env_id], 0, rb_props)

        # Apply damping randomization
        if "damping" in self.domain_rand_vec:
            damping_multipliers = self.domain_rand_vec["damping"][env_ids]
            for i, env_id in enumerate(env_ids):
                # Apply to all dofs
                dof_props = self.env.gym.get_actor_dof_properties(self.env.envs[env_id], 0)
                dof_props['damping'] *= damping_multipliers[i]
                self.env.gym.set_actor_dof_properties(self.env.envs[env_id], 0, dof_props)

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