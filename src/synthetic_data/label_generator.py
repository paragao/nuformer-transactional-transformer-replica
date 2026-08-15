"""Label generation for credit card activation prediction.

Creates time-delayed binary labels based on user personas and transaction
patterns. Users with certain behavioral signals are more likely to activate
a credit card within the 6-month prediction window.
"""

from __future__ import annotations

import numpy as np

from .schemas import UserPersona, Transaction, IncomeBracket, AgeGroup


class LabelGenerator:
    """Generate binary credit card activation labels.

    The label is: "Will this user activate a credit card within 6 months?"
    Probability is conditioned on:
    - Income bracket (higher income -> higher probability)
    - Credit score (higher score -> higher probability)
    - Digital affinity (more digital -> higher probability)
    - Transaction diversity (more diverse -> higher probability)
    - Spending discipline (moderate is best)
    - Age group (millennials highest)
    """

    def __init__(self, target_positive_rate: float = 0.15, seed: int = 42):
        self.target_rate = target_positive_rate
        self.rng = np.random.default_rng(seed)

    def generate_label(self, persona: UserPersona, transactions: list[Transaction]) -> bool:
        """Generate credit card activation label for a user."""
        # Compute activation probability from signals
        prob = self._compute_probability(persona, transactions)

        # Calibrate to target rate
        # We'll adjust the threshold post-hoc, but initial probabilities
        # are designed to center around target_rate
        return bool(self.rng.random() < prob)

    def generate_labels_batch(
        self, personas: list[UserPersona], transactions_by_user: dict[str, list[Transaction]]
    ) -> dict[str, bool]:
        """Generate labels for a batch of users, calibrated to target rate."""
        # First pass: compute raw probabilities
        probs = []
        for persona in personas:
            user_txns = transactions_by_user.get(persona.user_id, [])
            prob = self._compute_probability(persona, user_txns)
            probs.append(prob)

        probs = np.array(probs)

        # Calibrate: find threshold that gives target_rate
        # Sort probabilities and find cutoff
        sorted_probs = np.sort(probs)[::-1]
        n_positive = int(len(probs) * self.target_rate)
        if n_positive > 0 and n_positive < len(sorted_probs):
            threshold = sorted_probs[n_positive]
        else:
            threshold = 0.5

        # Add noise around threshold for soft boundary
        noise = self.rng.normal(0, 0.05, size=len(probs))
        labels = (probs + noise) > threshold

        return {persona.user_id: bool(label) for persona, label in zip(personas, labels)}

    def _compute_probability(self, persona: UserPersona, transactions: list[Transaction]) -> float:
        """Compute raw activation probability from user signals."""
        score = 0.0

        # Income signal (0 to 0.25)
        income_scores = {
            IncomeBracket.LOW: 0.05,
            IncomeBracket.MIDDLE: 0.15,
            IncomeBracket.HIGH: 0.22,
            IncomeBracket.PREMIUM: 0.20,  # slightly less - already have cards
        }
        score += income_scores.get(persona.income_bracket, 0.1)

        # Credit score signal (0 to 0.20)
        credit_norm = (persona.credit_score - 300) / 550
        score += 0.20 * credit_norm

        # Digital affinity (0 to 0.15)
        score += 0.15 * persona.digital_affinity

        # Age group signal (0 to 0.10)
        age_scores = {
            AgeGroup.GEN_Z: 0.08,
            AgeGroup.MILLENNIAL: 0.10,
            AgeGroup.GEN_X: 0.06,
            AgeGroup.BOOMER: 0.03,
        }
        score += age_scores.get(persona.age_group, 0.05)

        # Transaction diversity signal (0 to 0.15)
        if transactions:
            unique_categories = len(set(t.category for t in transactions))
            category_diversity = min(1.0, unique_categories / 8.0)
            score += 0.15 * category_diversity
        else:
            score += 0.02

        # Spending discipline (moderate is best) (0 to 0.10)
        # Bell curve centered at 0.5
        discipline_score = 1.0 - 4 * (persona.spending_discipline - 0.5) ** 2
        score += 0.10 * max(0, discipline_score)

        # Not already having a card (0 to 0.05)
        if not persona.has_credit_card:
            score += 0.05

        return float(np.clip(score, 0.01, 0.95))
