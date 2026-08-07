"""Compute 291-dimensional tabular features from transaction history.

Groups:
- Transaction aggregates (~100 features)
- Temporal patterns (~50 features)
- Merchant diversity (~30 features)
- Financial health (~40 features)
- Bureau/external (~30 features)
- Behavioral (~30 features)
- Demographics (~11 features)
"""

from __future__ import annotations

import numpy as np
from datetime import datetime, timedelta
from collections import Counter

from .schemas import Transaction, UserPersona, TransactionCategory


class TabularFeatureComputer:
    """Compute 291-dim tabular features from user transactions and persona."""

    NUM_FEATURES = 291

    def __init__(self, reference_date: str = "2024-01-01"):
        self.reference_date = datetime.fromisoformat(reference_date)

    def compute(self, persona: UserPersona, transactions: list[Transaction]) -> dict[str, float]:
        """Compute all 291 features for a single user."""
        features = {}

        # Filter to approved transactions before reference date
        txns = [t for t in transactions if t.status.value == "approved" and t.timestamp < self.reference_date]

        if not txns:
            return self._empty_features(persona)

        # Sort by time
        txns.sort(key=lambda t: t.timestamp)
        amounts = np.array([t.amount for t in txns])
        outflows = amounts[amounts < 0]
        inflows = amounts[amounts >= 0]

        # === Transaction Aggregates (~100 features) ===
        features.update(self._transaction_aggregates(txns, amounts, outflows, inflows))

        # === Temporal Patterns (~50 features) ===
        features.update(self._temporal_patterns(txns))

        # === Merchant Diversity (~30 features) ===
        features.update(self._merchant_diversity(txns))

        # === Financial Health (~40 features) ===
        features.update(self._financial_health(txns, persona, amounts))

        # === Bureau/External (~30 features) ===
        features.update(self._bureau_features(persona))

        # === Behavioral (~30 features) ===
        features.update(self._behavioral_features(txns, persona))

        # === Demographics (~11 features) ===
        features.update(self._demographic_features(persona))

        # Pad to 291 if needed
        features = self._pad_features(features)

        return features

    def _transaction_aggregates(
        self, txns: list[Transaction], amounts: np.ndarray, outflows: np.ndarray, inflows: np.ndarray
    ) -> dict[str, float]:
        """Transaction aggregate features."""
        f = {}

        # Time windows
        for days, suffix in [(30, "30d"), (60, "60d"), (90, "90d"), (180, "180d"), (365, "365d")]:
            cutoff = self.reference_date - timedelta(days=days)
            window_txns = [t for t in txns if t.timestamp >= cutoff]
            window_amts = np.array([t.amount for t in window_txns]) if window_txns else np.array([0.0])
            window_out = window_amts[window_amts < 0]
            window_in = window_amts[window_amts >= 0]

            f[f"total_spend_{suffix}"] = float(abs(window_out.sum())) if len(window_out) > 0 else 0.0
            f[f"total_income_{suffix}"] = float(window_in.sum()) if len(window_in) > 0 else 0.0
            f[f"txn_count_{suffix}"] = float(len(window_txns))
            f[f"avg_txn_amt_{suffix}"] = float(np.mean(np.abs(window_amts))) if len(window_amts) > 0 else 0.0
            f[f"max_txn_amt_{suffix}"] = float(np.max(np.abs(window_amts))) if len(window_amts) > 0 else 0.0
            f[f"std_txn_amt_{suffix}"] = float(np.std(window_amts)) if len(window_amts) > 1 else 0.0

        # Overall stats
        f["median_txn_amount"] = float(np.median(np.abs(amounts)))
        f["p25_txn_amount"] = float(np.percentile(np.abs(amounts), 25))
        f["p75_txn_amount"] = float(np.percentile(np.abs(amounts), 75))
        f["p95_txn_amount"] = float(np.percentile(np.abs(amounts), 95))
        f["iqr_txn_amount"] = f["p75_txn_amount"] - f["p25_txn_amount"]

        # Category shares (12 categories × 2 windows = 24 features)
        for cat in TransactionCategory:
            cat_txns_90 = [t for t in txns if t.category == cat and t.timestamp >= self.reference_date - timedelta(days=90)]
            f[f"cat_share_{cat.value}_90d"] = len(cat_txns_90) / max(1, len(txns))
            cat_txns_30 = [t for t in txns if t.category == cat and t.timestamp >= self.reference_date - timedelta(days=30)]
            f[f"cat_share_{cat.value}_30d"] = len(cat_txns_30) / max(1, len(txns))

        # Category amount stats (12 × 2 = 24 features)
        for cat in TransactionCategory:
            cat_amts = [abs(t.amount) for t in txns if t.category == cat]
            f[f"cat_avg_amt_{cat.value}"] = float(np.mean(cat_amts)) if cat_amts else 0.0
            f[f"cat_total_amt_{cat.value}"] = float(np.sum(cat_amts)) if cat_amts else 0.0

        return f

    def _temporal_patterns(self, txns: list[Transaction]) -> dict[str, float]:
        """Temporal pattern features."""
        f = {}

        hours = [t.timestamp.hour for t in txns]
        weekdays = [t.timestamp.weekday() for t in txns]

        # Time-of-day distribution
        f["morning_ratio"] = sum(1 for h in hours if 6 <= h < 12) / max(1, len(hours))
        f["afternoon_ratio"] = sum(1 for h in hours if 12 <= h < 18) / max(1, len(hours))
        f["evening_ratio"] = sum(1 for h in hours if 18 <= h < 23) / max(1, len(hours))
        f["night_ratio"] = sum(1 for h in hours if h < 6 or h >= 23) / max(1, len(hours))

        # Weekday vs weekend
        weekday_txns = [t for t in txns if t.timestamp.weekday() < 5]
        weekend_txns = [t for t in txns if t.timestamp.weekday() >= 5]
        f["weekday_ratio"] = len(weekday_txns) / max(1, len(txns))
        f["weekend_ratio"] = len(weekend_txns) / max(1, len(txns))

        weekday_spend = sum(abs(t.amount) for t in weekday_txns if t.amount < 0)
        weekend_spend = sum(abs(t.amount) for t in weekend_txns if t.amount < 0)
        f["weekend_spend_ratio"] = weekend_spend / max(1, weekday_spend + weekend_spend)

        # Inter-transaction time stats
        if len(txns) > 1:
            deltas = [(txns[i + 1].timestamp - txns[i].timestamp).total_seconds() / 3600
                      for i in range(len(txns) - 1)]
            f["avg_inter_txn_hours"] = float(np.mean(deltas))
            f["std_inter_txn_hours"] = float(np.std(deltas))
            f["max_inter_txn_hours"] = float(np.max(deltas))
            f["min_inter_txn_hours"] = float(np.min(deltas))
        else:
            f["avg_inter_txn_hours"] = 0.0
            f["std_inter_txn_hours"] = 0.0
            f["max_inter_txn_hours"] = 0.0
            f["min_inter_txn_hours"] = 0.0

        # Day-of-week distribution (7 features)
        day_counts = Counter(weekdays)
        for d in range(7):
            f[f"dow_{d}_ratio"] = day_counts.get(d, 0) / max(1, len(txns))

        # Month-of-year distribution (12 features)
        month_counts = Counter(t.timestamp.month for t in txns)
        for m in range(1, 13):
            f[f"month_{m}_ratio"] = month_counts.get(m, 0) / max(1, len(txns))

        # Recency
        f["days_since_last_txn"] = (self.reference_date - txns[-1].timestamp).days
        f["days_since_first_txn"] = (self.reference_date - txns[0].timestamp).days

        # Trend: compare last 30d vs previous 30d
        cutoff_30 = self.reference_date - timedelta(days=30)
        cutoff_60 = self.reference_date - timedelta(days=60)
        recent_count = sum(1 for t in txns if t.timestamp >= cutoff_30)
        prev_count = sum(1 for t in txns if cutoff_60 <= t.timestamp < cutoff_30)
        f["txn_count_trend"] = (recent_count - prev_count) / max(1, prev_count)

        return f

    def _merchant_diversity(self, txns: list[Transaction]) -> dict[str, float]:
        """Merchant diversity features."""
        f = {}

        merchants = [t.merchant_id for t in txns]
        merchant_counts = Counter(merchants)

        f["unique_merchants_total"] = float(len(merchant_counts))

        # Time-windowed
        for days, suffix in [(30, "30d"), (90, "90d")]:
            cutoff = self.reference_date - timedelta(days=days)
            window_merchants = [t.merchant_id for t in txns if t.timestamp >= cutoff]
            f[f"unique_merchants_{suffix}"] = float(len(set(window_merchants)))

        # Concentration (HHI)
        total = len(merchants)
        if total > 0:
            shares = np.array(list(merchant_counts.values())) / total
            f["merchant_hhi"] = float(np.sum(shares ** 2))
            f["top_merchant_share"] = float(shares.max())
            f["top3_merchant_share"] = float(np.sort(shares)[-3:].sum()) if len(shares) >= 3 else float(shares.sum())
        else:
            f["merchant_hhi"] = 0.0
            f["top_merchant_share"] = 0.0
            f["top3_merchant_share"] = 0.0

        # Category diversity (entropy)
        cat_counts = Counter(t.category.value for t in txns)
        if cat_counts:
            probs = np.array(list(cat_counts.values()), dtype=float)
            probs = probs / probs.sum()
            f["category_entropy"] = float(-np.sum(probs * np.log(probs + 1e-10)))
            f["num_active_categories"] = float(len(cat_counts))
        else:
            f["category_entropy"] = 0.0
            f["num_active_categories"] = 0.0

        # MCC diversity
        mcc_counts = Counter(t.merchant_category_code for t in txns)
        f["unique_mcc_codes"] = float(len(mcc_counts))

        return f

    def _financial_health(
        self, txns: list[Transaction], persona: UserPersona, amounts: np.ndarray
    ) -> dict[str, float]:
        """Financial health indicators."""
        f = {}

        outflows = amounts[amounts < 0]
        inflows = amounts[amounts >= 0]

        # Monthly averages
        months_active = max(1, (txns[-1].timestamp - txns[0].timestamp).days / 30)
        f["avg_monthly_spend"] = float(abs(outflows.sum()) / months_active) if len(outflows) > 0 else 0.0
        f["avg_monthly_income"] = float(inflows.sum() / months_active) if len(inflows) > 0 else 0.0

        # Savings rate
        total_income = inflows.sum() if len(inflows) > 0 else 1.0
        total_spend = abs(outflows.sum()) if len(outflows) > 0 else 0.0
        f["savings_rate"] = max(0, (total_income - total_spend) / max(1, total_income))

        # Credit utilization
        if persona.credit_limit > 0:
            credit_txns = [t for t in txns if t.transaction_type.value == "credit"]
            credit_balance = sum(abs(t.amount) for t in credit_txns[-30:])  # last 30 txns
            f["credit_utilization"] = min(1.0, credit_balance / persona.credit_limit)
        else:
            f["credit_utilization"] = 0.0

        # Debt-to-income
        f["debt_to_income"] = total_spend / max(1, total_income)

        # Balance volatility (std of daily net flow)
        daily_flows = {}
        for t in txns:
            day_key = t.timestamp.date()
            daily_flows[day_key] = daily_flows.get(day_key, 0) + t.amount
        if daily_flows:
            f["balance_volatility"] = float(np.std(list(daily_flows.values())))
        else:
            f["balance_volatility"] = 0.0

        # Overdraft proxy (days with net negative > income)
        f["high_spend_days"] = sum(1 for v in daily_flows.values() if v < -persona.monthly_income / 15)

        # Installment usage
        installment_txns = [t for t in txns if t.installments > 1]
        f["installment_ratio"] = len(installment_txns) / max(1, len(txns))
        f["avg_installments"] = float(np.mean([t.installments for t in installment_txns])) if installment_txns else 0.0

        # Spending stability (coefficient of variation of monthly spend)
        monthly_spends = {}
        for t in txns:
            if t.amount < 0:
                month_key = (t.timestamp.year, t.timestamp.month)
                monthly_spends[month_key] = monthly_spends.get(month_key, 0) + abs(t.amount)
        if len(monthly_spends) > 1:
            ms = list(monthly_spends.values())
            f["spend_cv"] = float(np.std(ms) / max(1, np.mean(ms)))
        else:
            f["spend_cv"] = 0.0

        return f

    def _bureau_features(self, persona: UserPersona) -> dict[str, float]:
        """Simulated bureau/external features."""
        f = {}
        rng = np.random.default_rng(hash(persona.user_id) % (2**32))

        f["credit_score"] = float(persona.credit_score)
        f["credit_score_band"] = float(persona.credit_score // 50)

        # Simulated bureau features (correlated with credit score)
        score_norm = (persona.credit_score - 300) / 550
        f["inquiry_count_6m"] = float(max(0, int(rng.poisson(3 * (1 - score_norm)))))
        f["inquiry_count_12m"] = f["inquiry_count_6m"] + float(max(0, int(rng.poisson(2 * (1 - score_norm)))))
        f["delinquency_flag"] = 1.0 if rng.random() < (0.3 * (1 - score_norm)) else 0.0
        f["delinquency_count"] = float(int(f["delinquency_flag"] * rng.integers(1, 4)))
        f["accounts_open"] = float(max(1, int(rng.normal(3 + score_norm * 4, 1.5))))
        f["accounts_closed"] = float(max(0, int(rng.poisson(1))))
        f["oldest_account_months"] = float(persona.account_age_days / 30)
        f["newest_account_months"] = float(max(1, persona.account_age_days / 30 - rng.integers(0, 24)))
        f["total_credit_lines"] = float(max(1, int(rng.normal(2 + score_norm * 3, 1))))
        f["total_balance"] = float(rng.lognormal(8 + score_norm * 2, 1))
        f["revolving_utilization"] = float(np.clip(rng.beta(2, 5 + score_norm * 5), 0, 1))
        f["public_records"] = float(max(0, int(rng.poisson(0.1 * (1 - score_norm)))))

        # Additional bureau noise features
        for i in range(18):
            f[f"bureau_feature_{i}"] = float(rng.normal(score_norm, 0.3))

        return f

    def _behavioral_features(self, txns: list[Transaction], persona: UserPersona) -> dict[str, float]:
        """Simulated behavioral features (app usage, support, etc.)."""
        f = {}
        rng = np.random.default_rng(hash(persona.user_id + "_behavior") % (2**32))

        # Digital engagement (correlated with digital_affinity)
        da = persona.digital_affinity
        f["app_sessions_30d"] = float(int(rng.poisson(20 * da + 5)))
        f["app_sessions_90d"] = f["app_sessions_30d"] * 3 + float(rng.integers(-5, 5))
        f["feature_usage_score"] = float(np.clip(rng.normal(da * 0.8, 0.15), 0, 1))
        f["notification_opt_in"] = 1.0 if rng.random() < da else 0.0
        f["biometric_enabled"] = 1.0 if rng.random() < da * 0.9 else 0.0

        # Support interactions
        f["support_tickets_6m"] = float(max(0, int(rng.poisson(0.8))))
        f["chat_interactions_6m"] = float(max(0, int(rng.poisson(1.5 * (1 - persona.spending_discipline)))))

        # Product engagement
        f["pix_keys_registered"] = float(min(5, int(rng.poisson(2 * da))))
        f["has_savings_account"] = 1.0 if persona.spending_discipline > 0.5 else 0.0
        f["has_investment"] = 1.0 if persona.income_bracket.value in ("high", "premium") and rng.random() < 0.4 else 0.0
        f["insurance_products"] = float(max(0, int(rng.poisson(0.3 * (persona.credit_score / 850)))))

        # Engagement trend
        f["engagement_trend_30d"] = float(rng.normal(0, 0.1))
        f["days_since_app_open"] = float(max(0, int(rng.exponential(3))))

        # Pad with noise features
        for i in range(16):
            f[f"behavior_feature_{i}"] = float(rng.normal(da * 0.5, 0.25))

        return f

    def _demographic_features(self, persona: UserPersona) -> dict[str, float]:
        """Demographic features from persona."""
        f = {}

        # One-hot encode categoricals
        for bracket in ["low", "middle", "high", "premium"]:
            f[f"income_bracket_{bracket}"] = 1.0 if persona.income_bracket.value == bracket else 0.0

        for age in ["gen_z", "millennial", "gen_x", "boomer"]:
            f[f"age_group_{age}"] = 1.0 if persona.age_group.value == age else 0.0

        f["account_tenure_months"] = float(persona.account_age_days / 30)
        f["is_primary_bank"] = 1.0 if persona.is_primary_bank else 0.0
        f["monthly_income_log"] = float(np.log1p(persona.monthly_income))

        return f

    def _empty_features(self, persona: UserPersona) -> dict[str, float]:
        """Return zero features for users with no transactions."""
        features = {f"feature_{i}": 0.0 for i in range(self.NUM_FEATURES)}
        features.update(self._demographic_features(persona))
        features.update(self._bureau_features(persona))
        return self._pad_features(features)

    def _pad_features(self, features: dict[str, float]) -> dict[str, float]:
        """Pad features to exactly NUM_FEATURES dimensions."""
        current = len(features)
        if current < self.NUM_FEATURES:
            for i in range(self.NUM_FEATURES - current):
                features[f"pad_feature_{i}"] = 0.0
        elif current > self.NUM_FEATURES:
            # Truncate (keep first NUM_FEATURES)
            keys = list(features.keys())[:self.NUM_FEATURES]
            features = {k: features[k] for k in keys}
        return features
