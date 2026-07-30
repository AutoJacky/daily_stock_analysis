# -*- coding: utf-8 -*-
"""Objective daily governance for the analysis model.

This service does not retrain an LLM and never submits broker orders.  It turns
persisted forward outcomes into a versioned, auditable state that can only keep
or reduce confidence on the next scheduled run.
"""

from __future__ import annotations

from datetime import date
import json
from typing import Any, Dict, Optional

from sqlalchemy import desc, select

from src.storage import AdaptiveLearningSnapshot, DatabaseManager, utc_naive_now


ADAPTIVE_LEARNING_POLICY_VERSION = "adaptive-governor-v1"
MIN_COMPLETED_OUTCOMES = 30
SHADOW_PROMOTION_OUTCOMES = 60


class AdaptiveLearningService:
    """Evaluate, persist, and retrieve the daily model-governance state."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager.get_instance()

    @classmethod
    def evaluate(
        cls,
        *,
        outcome_stats: Optional[Dict[str, Any]],
        backtest_summary: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        outcomes = outcome_stats if isinstance(outcome_stats, dict) else {}
        backtest = backtest_summary if isinstance(backtest_summary, dict) else {}

        total = cls._safe_int(outcomes.get("total"))
        completed = cls._safe_int(outcomes.get("completed"))
        unable = cls._safe_int(outcomes.get("unable"))
        hit_rate = cls._safe_float(outcomes.get("hit_rate_pct"))
        unable_rate = round(unable / total * 100, 2) if total else None

        backtest_total = cls._safe_int(backtest.get("total_evaluations"))
        backtest_accuracy = cls._ratio_or_pct(
            backtest.get("direction_accuracy_pct"),
            backtest.get("direction_accuracy"),
        )
        avg_simulated_return = cls._pct_value(
            backtest.get("avg_simulated_return_pct"),
            backtest.get("avg_return"),
        )

        reasons = []
        state = "collecting"
        confidence_factor = 1.0

        if completed < MIN_COMPLETED_OUTCOMES:
            reasons.append(
                f"有效后验样本{completed}个，未达到{MIN_COMPLETED_OUTCOMES}个治理门槛"
            )
        elif unable_rate is not None and unable_rate > 30:
            state = "data_blocked"
            confidence_factor = 0.65
            reasons.append(f"不可评估率{unable_rate:.1f}%高于30%，先修复数据覆盖")
        elif (
            (hit_rate is not None and hit_rate < 45)
            or (
                backtest_total >= MIN_COMPLETED_OUTCOMES
                and backtest_accuracy is not None
                and backtest_accuracy < 45
            )
            or (
                backtest_total >= MIN_COMPLETED_OUTCOMES
                and avg_simulated_return is not None
                and avg_simulated_return < -1
            )
        ):
            state = "restricted"
            confidence_factor = 0.65
            reasons.append("方向或模拟收益未达到最低稳定标准，自动进入限制状态")
        elif (
            hit_rate is None
            or hit_rate < 52
            or (
                backtest_total >= MIN_COMPLETED_OUTCOMES
                and backtest_accuracy is not None
                and backtest_accuracy < 50
            )
        ):
            state = "guarded"
            confidence_factor = 0.85
            reasons.append("样本已足够但稳定性仍一般，下一轮继续压低置信度")
        else:
            state = "stable"
            confidence_factor = 1.0
            reasons.append("客观后验达到当前稳定门槛，仅维持原置信度，不自动上调")

        shadow_profile = cls._select_shadow_champion(
            outcomes.get("profile_calibration")
        )
        if shadow_profile:
            reasons.append(
                f"{shadow_profile}档达到影子候选门槛，仅用于继续模拟验证"
            )

        return {
            "policy_version": ADAPTIVE_LEARNING_POLICY_VERSION,
            "state": state,
            "total_outcomes": total,
            "completed_outcomes": completed,
            "unable_outcomes": unable,
            "hit_rate_pct": hit_rate,
            "unable_rate_pct": unable_rate,
            "backtest_direction_accuracy_pct": backtest_accuracy,
            "avg_simulated_return_pct": avg_simulated_return,
            "confidence_factor": confidence_factor,
            "shadow_champion_profile": shadow_profile,
            # This invariant intentionally cannot be promoted by statistics.
            "live_trading_allowed": False,
            "reasons": reasons,
        }

    def run_daily(
        self,
        *,
        outcome_stats: Optional[Dict[str, Any]],
        backtest_summary: Optional[Dict[str, Any]],
        snapshot_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        decision = self.evaluate(
            outcome_stats=outcome_stats,
            backtest_summary=backtest_summary,
        )
        day = snapshot_date or date.today()
        now = utc_naive_now()
        values = {
            "snapshot_date": day,
            "scope": "global",
            "policy_version": decision["policy_version"],
            "state": decision["state"],
            "total_outcomes": decision["total_outcomes"],
            "completed_outcomes": decision["completed_outcomes"],
            "unable_outcomes": decision["unable_outcomes"],
            "hit_rate_pct": decision["hit_rate_pct"],
            "unable_rate_pct": decision["unable_rate_pct"],
            "backtest_direction_accuracy_pct": decision[
                "backtest_direction_accuracy_pct"
            ],
            "avg_simulated_return_pct": decision["avg_simulated_return_pct"],
            "confidence_factor": decision["confidence_factor"],
            "shadow_champion_profile": decision["shadow_champion_profile"],
            "live_trading_allowed": False,
            "reasons_json": json.dumps(
                decision["reasons"],
                ensure_ascii=False,
            ),
            "updated_at": now,
        }

        with self.db.get_session() as session:
            row = session.execute(
                select(AdaptiveLearningSnapshot)
                .where(
                    AdaptiveLearningSnapshot.snapshot_date == day,
                    AdaptiveLearningSnapshot.scope == "global",
                    AdaptiveLearningSnapshot.policy_version
                    == decision["policy_version"],
                )
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                row = AdaptiveLearningSnapshot(**values, created_at=now)
                session.add(row)
            else:
                for key, value in values.items():
                    setattr(row, key, value)
            session.commit()
            session.refresh(row)

        return self._serialize(row)

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        with self.db.get_session() as session:
            row = session.execute(
                select(AdaptiveLearningSnapshot)
                .where(
                    AdaptiveLearningSnapshot.scope == "global",
                    AdaptiveLearningSnapshot.policy_version
                    == ADAPTIVE_LEARNING_POLICY_VERSION,
                )
                .order_by(
                    desc(AdaptiveLearningSnapshot.snapshot_date),
                    desc(AdaptiveLearningSnapshot.id),
                )
                .limit(1)
            ).scalar_one_or_none()
            return self._serialize(row) if row is not None else None

    @classmethod
    def _select_shadow_champion(cls, calibration: Any) -> Optional[str]:
        if not isinstance(calibration, dict):
            return None
        breakdowns = calibration.get("breakdowns")
        if not isinstance(breakdowns, dict):
            return None
        buckets = breakdowns.get("decision_profile")
        if not isinstance(buckets, list):
            return None

        eligible = []
        for bucket in buckets:
            if not isinstance(bucket, dict):
                continue
            dimensions = bucket.get("dimensions")
            profile = (
                str(dimensions.get("decision_profile") or "").strip().lower()
                if isinstance(dimensions, dict)
                else ""
            )
            completed = cls._safe_int(bucket.get("completed"))
            hit_rate = cls._safe_float(bucket.get("hit_rate_pct"))
            unable_rate = cls._safe_float(bucket.get("unable_rate_pct"))
            if (
                profile in {"conservative", "balanced", "aggressive"}
                and completed >= SHADOW_PROMOTION_OUTCOMES
                and hit_rate is not None
                and hit_rate >= 55
                and (unable_rate is None or unable_rate <= 15)
            ):
                eligible.append((hit_rate, completed, profile))

        if not eligible:
            return None
        eligible.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return eligible[0][2]

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if result == result else None

    @classmethod
    def _ratio_or_pct(cls, pct_value: Any, ratio_value: Any) -> Optional[float]:
        direct = cls._safe_float(pct_value)
        if direct is not None:
            return direct
        ratio = cls._safe_float(ratio_value)
        if ratio is None:
            return None
        return ratio * 100 if abs(ratio) <= 1 else ratio

    @classmethod
    def _pct_value(cls, pct_value: Any, ratio_value: Any) -> Optional[float]:
        direct = cls._safe_float(pct_value)
        if direct is not None:
            return direct
        ratio = cls._safe_float(ratio_value)
        if ratio is None:
            return None
        return ratio * 100 if abs(ratio) <= 1 else ratio

    @staticmethod
    def _serialize(row: AdaptiveLearningSnapshot) -> Dict[str, Any]:
        try:
            reasons = json.loads(row.reasons_json or "[]")
        except (TypeError, ValueError):
            reasons = []
        return {
            "id": row.id,
            "snapshot_date": (
                row.snapshot_date.isoformat() if row.snapshot_date else None
            ),
            "scope": row.scope,
            "policy_version": row.policy_version,
            "state": row.state,
            "total_outcomes": row.total_outcomes,
            "completed_outcomes": row.completed_outcomes,
            "unable_outcomes": row.unable_outcomes,
            "hit_rate_pct": row.hit_rate_pct,
            "unable_rate_pct": row.unable_rate_pct,
            "backtest_direction_accuracy_pct": (
                row.backtest_direction_accuracy_pct
            ),
            "avg_simulated_return_pct": row.avg_simulated_return_pct,
            "confidence_factor": row.confidence_factor,
            "shadow_champion_profile": row.shadow_champion_profile,
            "live_trading_allowed": False,
            "reasons": reasons if isinstance(reasons, list) else [],
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
