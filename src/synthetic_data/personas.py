"""User persona generation with correlated attributes.

Generates diverse user personas with realistic correlations between
income, age, risk profile, credit score, and spending habits.
"""

from __future__ import annotations

import numpy as np

from .schemas import AgeGroup, IncomeBracket, RiskProfile, UserPersona


class PersonaGenerator:
    """Generate diverse user personas with correlated attributes."""

    # Income distributions by bracket (monthly, USD equivalent)
    INCOME_PARAMS = {
        IncomeBracket.LOW: (1200, 400),  # mean, std
        IncomeBracket.MIDDLE: (3500, 1000),
        IncomeBracket.HIGH: (8000, 2500),
        IncomeBracket.PREMIUM: (20000, 8000),
    }

    # Bracket probabilities by age group
    BRACKET_PROBS = {
        AgeGroup.GEN_Z: [0.45, 0.40, 0.12, 0.03],
        AgeGroup.MILLENNIAL: [0.20, 0.40, 0.30, 0.10],
        AgeGroup.GEN_X: [0.15, 0.30, 0.35, 0.20],
        AgeGroup.BOOMER: [0.25, 0.30, 0.25, 0.20],
    }

    # Age group probabilities (Nubank-like: younger skew)
    AGE_PROBS = [0.30, 0.40, 0.20, 0.10]

    # Risk profile probabilities by age
    RISK_PROBS = {
        AgeGroup.GEN_Z: [0.20, 0.40, 0.40],
        AgeGroup.MILLENNIAL: [0.25, 0.45, 0.30],
        AgeGroup.GEN_X: [0.40, 0.40, 0.20],
        AgeGroup.BOOMER: [0.55, 0.35, 0.10],
    }

    REGIONS = ["southeast", "south", "northeast", "north", "midwest"]
    REGION_PROBS = [0.42, 0.22, 0.20, 0.08, 0.08]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(self, num_users: int) -> list[UserPersona]:
        """Generate a batch of diverse user personas."""
        personas = []
        for i in range(num_users):
            persona = self._generate_single(f"user_{i:08d}")
            personas.append(persona)
        return personas

    def _generate_single(self, user_id: str) -> UserPersona:
        """Generate a single user persona with correlated attributes."""
        # Age group (use index to avoid numpy string coercion)
        age_groups = list(AgeGroup)
        age_idx = self.rng.choice(len(age_groups), p=self.AGE_PROBS)
        age_group = age_groups[age_idx]

        # Income bracket (correlated with age)
        bracket_probs = self.BRACKET_PROBS[age_group]
        brackets = list(IncomeBracket)
        bracket_idx = self.rng.choice(len(brackets), p=bracket_probs)
        income_bracket = brackets[bracket_idx]

        # Monthly income (from bracket distribution)
        mean, std = self.INCOME_PARAMS[income_bracket]
        monthly_income = max(500, self.rng.normal(mean, std))

        # Risk profile (correlated with age)
        risk_probs = self.RISK_PROBS[age_group]
        risk_profiles = list(RiskProfile)
        risk_idx = self.rng.choice(len(risk_profiles), p=risk_probs)
        risk_profile = risk_profiles[risk_idx]

        # Credit score (correlated with income and age)
        base_score = 580
        income_bonus = min(150, monthly_income / 200)
        age_bonus = {"gen_z": -30, "millennial": 20, "gen_x": 50, "boomer": 40}[age_group.value]
        noise = self.rng.normal(0, 40)
        credit_score = int(np.clip(base_score + income_bonus + age_bonus + noise, 300, 850))

        # Credit limit (correlated with income and credit score)
        credit_limit = monthly_income * (1.5 + (credit_score - 500) / 350) * self.rng.uniform(0.8, 1.2)
        credit_limit = max(0, min(100000, credit_limit))

        # Account age (correlated with age group)
        age_ranges = {
            AgeGroup.GEN_Z: (30, 730),
            AgeGroup.MILLENNIAL: (180, 1825),
            AgeGroup.GEN_X: (365, 2920),
            AgeGroup.BOOMER: (730, 3650),
        }
        min_age, max_age = age_ranges[age_group]
        account_age_days = int(self.rng.uniform(min_age, max_age))

        # Region
        region = self.rng.choice(self.REGIONS, p=self.REGION_PROBS)

        # Spending discipline (correlated with credit score and age)
        discipline_base = (credit_score - 300) / 550  # 0 to 1
        discipline_noise = self.rng.normal(0, 0.15)
        spending_discipline = float(np.clip(discipline_base + discipline_noise, 0, 1))

        # Digital affinity (correlated with age - younger = higher)
        digital_base = {"gen_z": 0.85, "millennial": 0.75, "gen_x": 0.55, "boomer": 0.35}[age_group.value]
        digital_noise = self.rng.normal(0, 0.12)
        digital_affinity = float(np.clip(digital_base + digital_noise, 0, 1))

        # Primary bank (higher income = more likely to have elsewhere)
        is_primary = self.rng.random() < (0.8 if income_bracket == IncomeBracket.LOW else 0.6)

        return UserPersona(
            user_id=user_id,
            income_bracket=income_bracket,
            age_group=age_group,
            risk_profile=risk_profile,
            monthly_income=round(monthly_income, 2),
            credit_limit=round(credit_limit, 2),
            credit_score=credit_score,
            account_age_days=account_age_days,
            is_primary_bank=is_primary,
            region=region,
            has_credit_card=False,
            spending_discipline=round(spending_discipline, 3),
            digital_affinity=round(digital_affinity, 3),
        )
