# Copyright 2026 Dmitri Manajev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk


@jaxls.Cost.create_factory
def limit_velocity_cost(
    vals: jaxls.VarValues,
    robot: pk.Robot,
    joint_var: jaxls.Var[jax.Array],
    prev_cfg: jax.Array,
    dt: float,
    weight: jax.Array | float,
) -> jax.Array:
    joint_vel = (vals[joint_var] - prev_cfg) / dt
    residual = jnp.maximum(0.0, jnp.abs(joint_vel) - robot.joints.velocity_limits)
    return (residual * weight).flatten()

# Same public shape as the other pyroki_snippets helpers.
def solve_ik_vel_cost(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz: np.ndarray,
    target_position: np.ndarray,
    prev_cfg: np.ndarray,
    dt: float = 0.01,
    pos_weight: float = 50.0,
    ori_weight: float = 10.0,
    limit_weight: float = 100.0,
    vel_weight: float = 0.1,
) -> np.ndarray:
    assert target_position.shape == (3,)
    assert target_wxyz.shape == (4,)

    target_link_index = robot.links.names.index(target_link_name)
    sol = _solve_ik_vel_cost_jax(
        robot=robot,
        target_link_index=jnp.array(target_link_index, dtype=jnp.int32),
        target_wxyz=jnp.array(target_wxyz, dtype=jnp.float32),
        target_position=jnp.array(target_position, dtype=jnp.float32),
        prev_cfg=jnp.array(prev_cfg, dtype=jnp.float32),
        dt=dt,
        pos_weight=pos_weight,
        ori_weight=ori_weight,
        limit_weight=limit_weight,
        vel_weight=vel_weight,
    )
    return np.array(sol)


@jdc.jit
def _solve_ik_vel_cost_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    prev_cfg: jax.Array,
    dt: jdc.Static[float] = 0.01,
    pos_weight: jdc.Static[float] = 50.0,
    ori_weight: jdc.Static[float] = 10.0,
    limit_weight: jdc.Static[float] = 100.0,
    vel_weight: jdc.Static[float] = 0.1,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_position
    )

    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            target_pose,
            target_link_index,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            weight=limit_weight,
        ),
        limit_velocity_cost(
            robot,
            joint_var,
            prev_cfg,
            dt,
            vel_weight,
        ),
    ]

    solution = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_cfg)]),
        )
    )
    return solution[joint_var]
