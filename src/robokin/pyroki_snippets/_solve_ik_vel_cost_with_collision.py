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

from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
from jax import Array
from jaxls import Cost, Var, VarValues


@Cost.create_factory
def limit_velocity_cost(
    vals: VarValues,
    robot: pk.Robot,
    joint_var: Var[Array],
    prev_cfg: Array,
    dt: float,
    weight: Array | float,
) -> Array:
    joint_vel = (vals[joint_var] - prev_cfg) / dt
    residual = jnp.maximum(0.0, jnp.abs(joint_vel) - robot.joints.velocity_limits)
    return (residual * weight).flatten()


def solve_ik_vel_cost_with_collision(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    target_position: np.ndarray,
    target_wxyz: np.ndarray,
    prev_cfg: np.ndarray,
    dt: float = 0.02,
    pos_weight: float = 5.0,
    ori_weight: float = 1.0,
    vel_weight: float = 0.1,
) -> np.ndarray:
    assert target_position.shape == (3,)
    assert target_wxyz.shape == (4,)
    target_link_idx = robot.links.names.index(target_link_name)
    cfg = _solve_ik_vel_cost_with_collision_jax(
        robot,
        coll,
        world_coll_list,
        jaxlie.SE3(jnp.concatenate([jnp.array(target_wxyz), jnp.array(target_position)], axis=-1)),
        jnp.array(target_link_idx),
        jnp.array(prev_cfg),
        dt,
        pos_weight,
        ori_weight,
        vel_weight,
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)
    return np.array(cfg)


@jdc.jit
def _solve_ik_vel_cost_with_collision_jax(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    T_world_target: jaxlie.SE3,
    target_link_index: jax.Array,
    prev_cfg: jax.Array,
    dt: jdc.Static[float],
    pos_weight: jdc.Static[float],
    ori_weight: jdc.Static[float],
    vel_weight: jdc.Static[float],
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    costs = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            target_pose=T_world_target,
            target_link_index=target_link_index,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose=jnp.array(prev_cfg),
            weight=0.01,
        ),
        pk.costs.self_collision_cost(
            robot,
            robot_coll=coll,
            joint_var=joint_var,
            margin=0.02,
            weight=5.0,
        ),
        limit_velocity_cost(
            robot,
            joint_var,
            prev_cfg,
            dt,
            vel_weight,
        ),
        pk.costs.limit_constraint(
            robot,
            joint_var,
        ),
    ]
    costs.extend(
        [
            pk.costs.world_collision_constraint(robot, coll, joint_var, world_coll, 0.05)
            for world_coll in world_coll_list
        ]
    )

    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_cfg)]),
        )
    )
    return sol[joint_var]
