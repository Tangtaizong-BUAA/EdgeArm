"""Semantic broad-push-face forecast guard for the V22 stock gripper.

The physical 96-part CAD-derived collision union is unchanged.  Eight frozen
parts are conditional contact candidates and the other 88 must retain the
same positive online planning margin.  Candidate penetration is accepted only
when the same-part MuJoCo contact exists and every contact on that part passes
the existing directed side-contact geometry checks.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .delayed_servo_acquisition_forecast_v1 import (
    DelayedServoAcquisitionForecastConfigV1,
    DelayedServoAcquisitionForecastV1,
)
from .sim2real_env_v10 import RealisticEdgeArmEnvV10, RealisticEnvV10Config
from .stock_gripper_action_guard_v3 import (
    StockGripperActionGuardConfigV3,
    StockGripperActionGuardV3,
)
from .stock_gripper_push_face_contact_v22 import (
    STOCK_GRIPPER_PUSH_FACE_CONTACT_FORMAT_V22,
    stock_gripper_push_face_profile_v22,
)


STOCK_GRIPPER_FORECAST_FORMAT_V4 = "edgearm-v22-stock-gripper-semantic-push-face-servo-forecast-v4"
STOCK_GRIPPER_ACTION_GUARD_FORMAT_V4 = (
    "edgearm-v22-stock-gripper-recursive-zero-invalid-contact-guard-v4.2"
)


class StockGripperDelayedServoForecastV4(DelayedServoAcquisitionForecastV1):
    """Exact V10 forecast over the V22 96/8/88 geometry identity."""

    def __init__(
        self,
        env: RealisticEdgeArmEnvV10,
        config: DelayedServoAcquisitionForecastConfigV1 | None = None,
    ) -> None:
        if type(env) is not RealisticEdgeArmEnvV10:
            raise TypeError("stock-gripper forecast V4 requires exact V10")
        if config is not None and type(config) is not DelayedServoAcquisitionForecastConfigV1:
            raise TypeError("stock-gripper forecast V4 requires exact config")
        profile = stock_gripper_push_face_profile_v22(env)
        if profile["format"] != STOCK_GRIPPER_PUSH_FACE_CONTACT_FORMAT_V22:
            raise RuntimeError("stock-gripper forecast V4 lost V22 profile")
        if (
            profile["safety_union_geom_count"] != 96
            or profile["contact_candidate_geom_count"] != 8
            or profile["safety_only_geom_count"] != 88
        ):
            raise RuntimeError("stock-gripper forecast V4 requires exact 96/8/88")

        self.env = env
        self.config = config or DelayedServoAcquisitionForecastConfigV1()
        self._safety_geoms = tuple(int(value) for value in env._ids["tool_safety_geoms"])
        self._contact_geoms = tuple(int(value) for value in env._ids["tool_contact_geoms"])
        self._contact_roles = tuple(str(value) for value in env._ids["tool_contact_geom_roles"])
        if len(self._contact_roles) != len(self._contact_geoms) or len(set(self._contact_roles)) != len(
            self._contact_roles
        ):
            raise RuntimeError("V4 contact roles are incomplete or duplicated")
        self._contact_columns = np.asarray(
            [self._safety_geoms.index(value) for value in self._contact_geoms],
            dtype=np.int64,
        )
        contact_set = set(self._contact_geoms)
        self._safety_only_geoms = tuple(value for value in self._safety_geoms if value not in contact_set)
        self._safety_only_columns = np.asarray(
            [self._safety_geoms.index(value) for value in self._safety_only_geoms],
            dtype=np.int64,
        )
        if len(self._safety_only_geoms) != 88:
            raise RuntimeError("V4 forecast lost 88 safety-only parts")
        self._penetration_tolerance_m = float(env.contact_feasible_config.reset_penetration_tolerance_m)

    def audit_trace(self, trace: Mapping[str, Any]) -> dict[str, Any]:
        """Make the online guard match the zero-invalid-contact data gate."""

        report = super().audit_trace(trace)
        invalid = np.asarray(
            trace.get("invalid_tool_block_contact_count_by_role", ()),
            dtype=np.int64,
        )
        expected = (
            int(trace.get("physics_substeps", -1)),
            len(self._contact_roles),
        )
        if invalid.shape != expected or np.any(invalid < 0):
            raise RuntimeError("V4 strict contact audit lost its role matrix")
        invalid_total = int(np.sum(invalid))
        invalid_substeps = np.flatnonzero(np.any(invalid > 0, axis=1))
        report.update(
            {
                "strict_zero_invalid_contact_required": True,
                "invalid_tool_block_contact_count": invalid_total,
                "invalid_tool_block_contact_substeps": int(
                    len(invalid_substeps)
                ),
                "first_invalid_tool_block_contact": (
                    {}
                    if not len(invalid_substeps)
                    else {
                        "substep_index": int(invalid_substeps[0]),
                        "count_by_role": {
                            role: int(
                                invalid[int(invalid_substeps[0]), role_index]
                            )
                            for role_index, role in enumerate(
                                self._contact_roles
                            )
                            if invalid[
                                int(invalid_substeps[0]), role_index
                            ]
                            > 0
                        },
                    }
                ),
                "valid": bool(report["valid"] and invalid_total == 0),
            }
        )
        return report

    def forecast(self, candidate_action: np.ndarray) -> dict[str, Any]:
        report = super().forecast(candidate_action)
        report["format"] = STOCK_GRIPPER_FORECAST_FORMAT_V4
        report["contact_identity_format"] = STOCK_GRIPPER_PUSH_FACE_CONTACT_FORMAT_V22
        report["contact_candidate_geom_count"] = len(self._contact_geoms)
        report["safety_only_geom_count"] = len(self._safety_only_geoms)
        report["historical_94_field_names_emitted"] = False
        return report


class StockGripperActionGuardV4(StockGripperActionGuardV3):
    """V3 latched-target transport with V22 semantic contact authority."""

    guard_format = STOCK_GRIPPER_ACTION_GUARD_FORMAT_V4

    def __init__(
        self,
        env: RealisticEdgeArmEnvV10,
        config: StockGripperActionGuardConfigV3 | None = None,
    ) -> None:
        selected = config or StockGripperActionGuardConfigV3()
        if type(selected) is not StockGripperActionGuardConfigV3:
            raise TypeError("stock-gripper action guard V4 requires exact V3 config")
        if type(env) is not RealisticEdgeArmEnvV10:
            raise TypeError("stock-gripper action guard V4 requires exact V10")
        if type(env.config) is not RealisticEnvV10Config:
            raise TypeError("stock-gripper action guard V4 requires exact V10 config")
        if (
            float(env.config.command_loss_probability) > 0.0
            or float(env.config.command_burst_start_probability) > 0.0
        ):
            raise RuntimeError("guard V4 initial curriculum requires deterministic transport")
        self.env = env
        self.config = selected
        self.forecast = StockGripperDelayedServoForecastV4(
            env,
            DelayedServoAcquisitionForecastConfigV1(
                minimum_safety_only_block_clearance_m=(selected.hard_executed_safety_only_clearance_m)
            ),
        )


__all__ = [
    "STOCK_GRIPPER_ACTION_GUARD_FORMAT_V4",
    "STOCK_GRIPPER_FORECAST_FORMAT_V4",
    "StockGripperActionGuardV4",
    "StockGripperDelayedServoForecastV4",
]
