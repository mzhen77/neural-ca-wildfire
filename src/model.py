"""CNN-parameterized Probabilistic Cellular Automaton for wildfire spread.

Architecture:
- FuelEmbedding: Learnable embedding for categorical FBFM40 fuel model indices.
- MSCNNParameterGenerator: Multi-scale CNN that produces 6 spatial parameter maps
  from environmental inputs + fuel embedding. Kernel sizes 3, 5, ..., K.
- WildfireCA: Stateless CA physics engine using the 3-state model (Eqs 5-10).
  Fuel contribution is learned via CNN fuel_factor instead of hard-coded canopy modifiers.
- WildfireModel: Wraps FuelEmbedding + CNN + CA for end-to-end differentiable simulation.
"""

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp


# Direction offsets for 8-neighbor CA (row, col)
# Order: N, NE, E, SE, S, SW, W, NW
NEIGHBOR_OFFSETS = (
    (-1, 0),   # N
    (-1, 1),   # NE
    (0, 1),    # E
    (1, 1),    # SE
    (1, 0),    # S
    (1, -1),   # SW
    (0, -1),   # W
    (-1, -1),  # NW
)

# Direction FROM neighbor TO target (fire spread direction)
SPREAD_ANGLES = (
    -jnp.pi / 2,        # N neighbor → fire spreads S
    -3 * jnp.pi / 4,    # NE → SW
    jnp.pi,             # E → W
    3 * jnp.pi / 4,     # SE → NW
    jnp.pi / 2,         # S → N
    jnp.pi / 4,         # SW → NE
    0.0,                 # W → E
    -jnp.pi / 4,        # NW → SE
)


def compute_wind_speed_direction(wind_u, wind_v):
    """Convert wind components to speed and direction.

    Returns: (speed, direction) where direction is radians (0=East, π/2=North).
    """
    speed = jnp.sqrt(wind_u**2 + wind_v**2)
    direction = jnp.arctan2(wind_v, wind_u) % (2 * jnp.pi)
    return speed, direction


class FuelEmbedding(eqx.Module):
    """Learnable embedding for categorical FBFM40 fuel model indices."""

    embedding: jnp.ndarray  # (num_classes, embed_dim)

    def __init__(self, config, key):
        num_classes = config['preprocessing']['fbfm_num_classes']
        embed_dim = config['preprocessing']['fbfm_embedding_dim']
        self.embedding = jax.random.normal(key, (num_classes, embed_dim)) * 0.1

    def __call__(self, fuel_type_map):
        """Map (H, W) integer indices to (embed_dim, H, W) dense vectors."""
        embedded = self.embedding[fuel_type_map]  # (H, W, embed_dim)
        return jnp.transpose(embedded, (2, 0, 1))  # (embed_dim, H, W)


class MSCNNParameterGenerator(eqx.Module):
    """Multi-Scale CNN that generates 6 spatial CA parameter maps.

    Parallel branches with kernels k=3, 5, ..., K, concatenated and
    compressed via 1x1 pointwise convolution to 6 output channels.
    Dropout applied after branch activations for regularization.
    """

    conv_branches: list
    conv_out: eqx.nn.Conv2d
    dropout: eqx.nn.Dropout

    def __init__(self, config, key):
        in_channels = config['model']['cnn_in_channels']
        kernel_size = config['model']['kernel_size']
        branch_out = config['model']['branch_out_channels']
        n_params = config['model']['n_ca_params']
        dropout_rate = config['model'].get('dropout_rate', 0.0)

        self.conv_branches = []
        n_branches = max(1, (kernel_size - 1) // 2)
        keys = jax.random.split(key, n_branches + 1)

        for i, k in enumerate(range(3, kernel_size + 1, 2)):
            padding = k // 2  # SAME padding
            branch = eqx.nn.Conv2d(
                in_channels, branch_out, kernel_size=k,
                padding=padding, key=keys[i],
            )
            self.conv_branches.append(branch)

        concat_channels = branch_out * len(self.conv_branches)
        self.conv_out = eqx.nn.Conv2d(
            concat_channels, n_params, kernel_size=1, key=keys[-1],
        )
        self.dropout = eqx.nn.Dropout(p=dropout_rate)

    def __call__(self, x, *, key=None):
        """Generate 6 CA parameter maps from input tensor.

        Args:
            x: (C, H, W) environmental input tensor (13 channels)
            key: PRNG key for dropout (required during training)

        Returns:
            Dict with: p_base, alpha_w1, alpha_w2, alpha_s, alpha_gamma, fuel_factor
        """
        branch_outputs = [jax.nn.relu(branch(x)) for branch in self.conv_branches]

        if len(branch_outputs) > 1:
            h_concat = jnp.concatenate(branch_outputs, axis=0)
        else:
            h_concat = branch_outputs[0]

        h_concat = self.dropout(h_concat, key=key)
        raw_params = self.conv_out(h_concat)  # (6, H, W)

        p_base = jax.nn.sigmoid(raw_params[0])        # (H, W) in (0, 1)
        alpha_w1 = jax.nn.softplus(raw_params[1])     # (H, W) > 0
        alpha_w2 = jax.nn.softplus(raw_params[2])     # (H, W) > 0
        alpha_s = jax.nn.softplus(raw_params[3])       # (H, W) > 0
        alpha_gamma = jax.nn.softplus(raw_params[4])  # (H, W) > 0
        fuel_factor = jax.nn.softplus(raw_params[5])  # (H, W) > 0, learned fuel contribution

        return {
            'p_base': p_base,
            'alpha_w1': alpha_w1,
            'alpha_w2': alpha_w2,
            'alpha_s': alpha_s,
            'alpha_gamma': alpha_gamma,
            'fuel_factor': fuel_factor,
        }


class WildfireCA:
    """Stateless CA physics engine for 3-state fire spread model.

    No learnable parameters. All CA parameters come from CNN-generated spatial maps.
    Fuel contribution is via learned fuel_factor instead of hard-coded canopy modifiers.
    """

    @staticmethod
    def compute_wind_factor(alpha_w1, alpha_w2, wind_speed, wind_dir, spread_angle):
        """Wind influence factor.

        κ_wind = exp(α_w1 * V) * exp(α_w2 * V * [cos(spread_angle - wind_dir) - 1])
        """
        wind_speed_clipped = jnp.clip(wind_speed, 0.0, 50.0)

        exp_arg1 = jnp.clip(alpha_w1 * wind_speed_clipped, -20.0, 20.0)
        speed_factor = jnp.exp(exp_arg1)

        alignment = jnp.cos(spread_angle - wind_dir) - 1.0
        exp_arg2 = jnp.clip(alpha_w2 * wind_speed_clipped * alignment, -20.0, 20.0)
        direction_factor = jnp.exp(exp_arg2)

        return speed_factor * direction_factor

    @staticmethod
    def compute_slope_factor(alpha_s, directional_slope):
        """Slope influence factor: κ_slope = exp(α_s * S_dir)."""
        exp_arg = jnp.clip(alpha_s * directional_slope, -20.0, 20.0)
        return jnp.exp(exp_arg)

    @staticmethod
    def compute_ignition_probability(
        fire_intensity, ca_params,
        wind_speed, wind_dir, slope, aspect,
    ):
        """Compute ignition probability with CNN-learned fuel factor.

        φ = p_base * fuel_factor_neighbor * κ_wind * κ_slope
        γ = α_γ * fuel_factor_target
        p_ignite = 1 - exp(-γ * λ)
        """
        ny, nx = fire_intensity.shape
        p_base = ca_params['p_base']
        alpha_w1 = ca_params['alpha_w1']
        alpha_w2 = ca_params['alpha_w2']
        alpha_s = ca_params['alpha_s']
        alpha_gamma = ca_params['alpha_gamma']
        fuel_factor = ca_params['fuel_factor']

        fire_pad = jnp.pad(fire_intensity, 1, mode='constant', constant_values=0.0)
        fuel_pad = jnp.pad(fuel_factor, 1, mode='edge')
        speed_pad = jnp.pad(wind_speed, 1, mode='edge')
        dir_pad = jnp.pad(wind_dir, 1, mode='edge')
        slope_pad = jnp.pad(slope, 1, mode='edge')
        aspect_pad = jnp.pad(aspect, 1, mode='edge')

        total_lambda = jnp.zeros((ny, nx))

        for i, (dr, dc) in enumerate(NEIGHBOR_OFFSETS):
            r_start, r_end = 1 + dr, ny + 1 + dr
            c_start, c_end = 1 + dc, nx + 1 + dc

            neighbor_fire = fire_pad[r_start:r_end, c_start:c_end]
            neighbor_fuel = fuel_pad[r_start:r_end, c_start:c_end]
            neighbor_speed = speed_pad[r_start:r_end, c_start:c_end]
            neighbor_dir = dir_pad[r_start:r_end, c_start:c_end]
            neighbor_slope = slope_pad[r_start:r_end, c_start:c_end]
            neighbor_aspect = aspect_pad[r_start:r_end, c_start:c_end]

            spread_angle = SPREAD_ANGLES[i]
            dir_slope = neighbor_slope * jnp.cos(spread_angle - neighbor_aspect)

            wind_factor = WildfireCA.compute_wind_factor(
                alpha_w1, alpha_w2, neighbor_speed, neighbor_dir, spread_angle
            )
            slope_factor = WildfireCA.compute_slope_factor(alpha_s, dir_slope)

            phi = jnp.clip(
                p_base * neighbor_fuel * wind_factor * slope_factor,
                0.0, 100.0,
            )
            total_lambda = total_lambda + neighbor_fire * phi

        # Target susceptibility: γ = α_γ * fuel_factor_target
        gamma = jnp.clip(alpha_gamma * fuel_factor, 0.0, 100.0)

        exp_arg = jnp.clip(-gamma * total_lambda, -50.0, 50.0)
        p_ignite = 1.0 - jnp.exp(exp_arg)
        return jnp.clip(p_ignite, 0.0, 1.0)

    @staticmethod
    def step_deterministic(
        p_unburned, p_burning, p_burned,
        ca_params,
        wind_speed, wind_dir, slope, aspect,
        fuel_mask, burn_duration,
    ):
        """One deterministic 3-state CA step.

        Uses stop_gradient(p_burning) as fire intensity input for gradient stability.

        Returns: (new_p_unburned, new_p_burning, new_p_burned)
        """
        # stop_gradient on p_burning prevents gradient explosion
        pb_detached = jax.lax.stop_gradient(p_burning)

        p_ignite = WildfireCA.compute_ignition_probability(
            pb_detached, ca_params,
            wind_speed, wind_dir, slope, aspect,
        )

        # Fuel mask from FBFM40: suppress ignition on non-burnable cells
        p_ignite = p_ignite * fuel_mask

        # Continuation probability
        p_continue = 1.0 - 1.0 / burn_duration

        # 3-state transitions (Eqs 5-7)
        newly_burning = p_unburned * p_ignite
        burned_now = p_burning * (1.0 - p_continue)

        # State updates (Eqs 8-10)
        new_p_unburned = jnp.clip(p_unburned - newly_burning, 0.0, 1.0)
        new_p_burning = jnp.clip(p_burning + newly_burning - burned_now, 0.0, 1.0)
        new_p_burned = jnp.clip(p_burned + burned_now, 0.0, 1.0)

        return new_p_unburned, new_p_burning, new_p_burned


class WildfireModel(eqx.Module):
    """End-to-end differentiable wildfire spread model.

    Wraps FuelEmbedding + MSCNNParameterGenerator + WildfireCA for training.
    """

    fuel_embedding: FuelEmbedding
    cnn: MSCNNParameterGenerator
    burn_duration: int = eqx.field(static=True)

    def __init__(self, config, key):
        key_embed, key_cnn = jax.random.split(key)
        self.fuel_embedding = FuelEmbedding(config, key_embed)
        self.cnn = MSCNNParameterGenerator(config, key_cnn)
        self.burn_duration = config['model']['burn_duration']

    def generate_params(self, static_cnn_input, wind_u_day, wind_v_day, fuel_type_map, *, key=None):
        """Generate spatial CA parameters for a single day.

        Concatenates static CNN channels + fuel embedding + daily wind → CNN → 6 param maps.

        Args:
            static_cnn_input: (7, H, W) static environmental channels
            wind_u_day: (H, W) wind u for this day
            wind_v_day: (H, W) wind v for this day
            fuel_type_map: (H, W) int32 FBFM40 indices
            key: PRNG key for dropout (None during inference)

        Returns:
            dict with p_base, alpha_w1, alpha_w2, alpha_s, alpha_gamma, fuel_factor
        """
        fuel_embed = self.fuel_embedding(fuel_type_map)  # (4, H, W)
        wind_stack = jnp.stack([wind_u_day, wind_v_day], axis=0)  # (2, H, W)
        cnn_input = jnp.concatenate([static_cnn_input, fuel_embed, wind_stack], axis=0)  # (13, H, W)
        return self.cnn(cnn_input, key=key)
