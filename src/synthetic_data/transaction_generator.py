"""Core transaction generator with realistic statistical distributions.

Implements category-specific amount distributions, temporal patterns,
recurring transactions, and seasonal effects matching real banking data.
"""

from __future__ import annotations

import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from .schemas import (
    DatasetConfig,
    Transaction,
    TransactionCategory,
    TransactionStatus,
    TransactionType,
    UserPersona,
    IncomeBracket,
)


class TransactionGenerator:
    """Generate realistic financial transactions per user persona.

    Produces transactions with:
    - Category-specific amount distributions (log-normal/normal)
    - Temporal patterns (weekday/weekend bias, time-of-day)
    - Recurring transactions (salary, rent, utilities, subscriptions)
    - Seasonal multipliers (holiday spending)
    - Persona-conditioned spending levels
    """

    # Category configs: (amount_distribution, mean, std, type, is_inflow)
    # For log-normal: mean/std are of the underlying normal (log-space)
    CATEGORY_PARAMS = {
        TransactionCategory.SALARY: ("normal", 4500, 1500, TransactionType.TRANSFER, True),
        TransactionCategory.RENT: ("normal", 1500, 500, TransactionType.BOLETO, False),
        TransactionCategory.GROCERIES: ("lognormal", 4.17, 0.5, TransactionType.DEBIT, False),
        TransactionCategory.DINING: ("lognormal", 3.56, 0.6, TransactionType.DEBIT, False),
        TransactionCategory.COFFEE: ("normal", 5.5, 1.5, TransactionType.DEBIT, False),
        TransactionCategory.TRANSPORT: ("lognormal", 3.22, 0.7, TransactionType.DEBIT, False),
        TransactionCategory.SHOPPING: ("lognormal", 4.32, 0.8, TransactionType.CREDIT, False),
        TransactionCategory.UTILITIES: ("normal", 150, 50, TransactionType.BOLETO, False),
        TransactionCategory.SUBSCRIPTIONS: ("normal", 15, 8, TransactionType.DEBIT, False),
        TransactionCategory.HEALTHCARE: ("lognormal", 4.79, 1.0, TransactionType.CREDIT, False),
        TransactionCategory.ENTERTAINMENT: ("lognormal", 3.22, 0.7, TransactionType.DEBIT, False),
        TransactionCategory.TRANSFERS: ("lognormal", 6.21, 1.2, TransactionType.PIX, False),
    }

    # Monthly frequency ranges per category (min, max transactions/month)
    FREQUENCY_RANGES = {
        TransactionCategory.SALARY: (2, 2),  # fixed: 1st and 15th
        TransactionCategory.RENT: (1, 1),  # fixed: 1st
        TransactionCategory.GROCERIES: (8, 16),
        TransactionCategory.DINING: (8, 20),
        TransactionCategory.COFFEE: (15, 22),  # ~weekdays
        TransactionCategory.TRANSPORT: (15, 25),
        TransactionCategory.SHOPPING: (3, 8),
        TransactionCategory.UTILITIES: (1, 3),  # fixed: 15th
        TransactionCategory.SUBSCRIPTIONS: (2, 5),
        TransactionCategory.HEALTHCARE: (0, 2),
        TransactionCategory.ENTERTAINMENT: (3, 8),
        TransactionCategory.TRANSFERS: (2, 8),
    }

    # Seasonal multipliers by month (1-indexed)
    SEASONAL_MULTIPLIERS = {
        1: 0.85, 2: 0.95, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0,
        7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.15, 12: 1.30,
    }

    # Merchant pools per category
    MERCHANTS = {
        TransactionCategory.SALARY: ["EMPLOYER_DIRECT_DEPOSIT", "CONTRACT_PAYMENT", "FREELANCE_INCOME"],
        TransactionCategory.RENT: ["LANDLORD_PAYMENT", "REALESTATE_MGMT", "HOUSING_COOP"],
        TransactionCategory.GROCERIES: [
            "WALMART", "CARREFOUR", "EXTRA_HIPER", "PAO_DE_ACUCAR", "ASSAI",
            "ATACADAO", "NATURAL_MARKET", "HORTIFRUTI", "MERCADO_LIVRE_FOOD",
        ],
        TransactionCategory.DINING: [
            "IFOOD", "RAPPI", "UBER_EATS", "MCDONALDS", "BURGER_KING",
            "SUBWAY", "PIZZA_HUT", "STARBUCKS", "LOCAL_RESTAURANT",
            "OUTBACK", "MADERO", "SUSHI_DELIVERY",
        ],
        TransactionCategory.COFFEE: [
            "STARBUCKS", "LOCAL_CAFE", "TIM_HORTONS", "DUNKIN",
            "PADARIA_LOCAL", "CAFE_CULTURA", "COFFEE_BEAN",
        ],
        TransactionCategory.TRANSPORT: [
            "UBER", "99_TAXI", "LYFT", "GAS_STATION_SHELL", "GAS_STATION_BR",
            "METRO_CARD", "BUS_PASS", "TOLL_AUTOPASS", "PARKING_ESTAPAR",
        ],
        TransactionCategory.SHOPPING: [
            "AMAZON", "MERCADO_LIVRE", "SHOPEE", "MAGAZINE_LUIZA",
            "AMERICANAS", "RENNER", "ZARA", "NIKE", "APPLE_STORE",
            "ALIEXPRESS", "SHEIN", "CASAS_BAHIA",
        ],
        TransactionCategory.UTILITIES: [
            "ENEL_ENERGY", "SABESP_WATER", "COMGAS", "INTERNET_VIVO",
            "INTERNET_CLARO", "PHONE_TIM",
        ],
        TransactionCategory.SUBSCRIPTIONS: [
            "NETFLIX", "SPOTIFY", "AMAZON_PRIME", "DISNEY_PLUS",
            "HBO_MAX", "YOUTUBE_PREMIUM", "APPLE_MUSIC", "GYM_SMARTFIT",
            "CLOUD_STORAGE", "NEWS_SUBSCRIPTION",
        ],
        TransactionCategory.HEALTHCARE: [
            "FARMACIA_DROGASIL", "FARMACIA_PAGUE_MENOS", "HOSPITAL_CLINIC",
            "DENTIST_OFFICE", "LAB_EXAMS", "HEALTH_INSURANCE",
        ],
        TransactionCategory.ENTERTAINMENT: [
            "CINEMA_CINEMARK", "THEATER_TICKET", "CONCERT_TICKET",
            "STEAM_GAMES", "PLAYSTATION_STORE", "BOOK_STORE",
            "PARK_ADMISSION", "MUSEUM_TICKET",
        ],
        TransactionCategory.TRANSFERS: [
            "PIX_TRANSFER", "TED_TRANSFER", "DOC_TRANSFER",
            "SPLIT_BILL", "FRIEND_PAYMENT",
        ],
    }

    # MCC codes by category
    MCC_CODES = {
        TransactionCategory.SALARY: "6012",
        TransactionCategory.RENT: "6513",
        TransactionCategory.GROCERIES: "5411",
        TransactionCategory.DINING: "5812",
        TransactionCategory.COFFEE: "5814",
        TransactionCategory.TRANSPORT: "4121",
        TransactionCategory.SHOPPING: "5311",
        TransactionCategory.UTILITIES: "4900",
        TransactionCategory.SUBSCRIPTIONS: "4899",
        TransactionCategory.HEALTHCARE: "8099",
        TransactionCategory.ENTERTAINMENT: "7832",
        TransactionCategory.TRANSFERS: "6012",
    }

    def __init__(self, config: DatasetConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.start_date = datetime.fromisoformat(config.start_date)
        self.end_date = datetime.fromisoformat(config.end_date)
        self.total_days = (self.end_date - self.start_date).days

    def generate_for_user(self, persona: UserPersona) -> list[Transaction]:
        """Generate all transactions for a single user persona."""
        transactions = []

        # Income multiplier based on persona
        income_mult = persona.monthly_income / 4500.0  # normalized to median

        # Determine number of transactions
        num_months = self.total_days / 30.0
        base_monthly_txns = self.rng.integers(30, 80)
        total_txns = int(base_monthly_txns * num_months)
        total_txns = np.clip(
            total_txns,
            self.config.min_transactions_per_user,
            self.config.max_transactions_per_user,
        )

        # Generate recurring transactions first
        transactions.extend(self._generate_recurring(persona, income_mult))

        # Generate variable transactions
        transactions.extend(
            self._generate_variable(persona, income_mult, total_txns - len(transactions))
        )

        # Sort by timestamp
        transactions.sort(key=lambda t: t.timestamp)
        return transactions

    def _generate_recurring(self, persona: UserPersona, income_mult: float) -> list[Transaction]:
        """Generate recurring transactions (salary, rent, utilities, subscriptions)."""
        transactions = []
        current_date = self.start_date

        while current_date < self.end_date:
            month_start = current_date.replace(day=1)
            month = current_date.month

            # Salary (1st and 15th)
            for day in [1, 15]:
                txn_date = month_start.replace(day=day)
                if self.start_date <= txn_date < self.end_date:
                    amount = self._sample_amount(TransactionCategory.SALARY, income_mult)
                    transactions.append(self._make_transaction(
                        persona.user_id, txn_date, amount, TransactionCategory.SALARY
                    ))

            # Rent (1st)
            if persona.monthly_income > 1500:  # only if income supports it
                txn_date = month_start.replace(day=1)
                if self.start_date <= txn_date < self.end_date:
                    amount = self._sample_amount(TransactionCategory.RENT, income_mult * 0.3)
                    transactions.append(self._make_transaction(
                        persona.user_id, txn_date, -amount, TransactionCategory.RENT
                    ))

            # Utilities (15th)
            txn_date = month_start.replace(day=15)
            if self.start_date <= txn_date < self.end_date:
                amount = self._sample_amount(TransactionCategory.UTILITIES, 1.0)
                transactions.append(self._make_transaction(
                    persona.user_id, txn_date, -amount, TransactionCategory.UTILITIES
                ))

            # Subscriptions (variable day, 2-5 per month)
            n_subs = self.rng.integers(2, min(6, int(3 + persona.digital_affinity * 4)))
            for _ in range(n_subs):
                day = self.rng.integers(1, 29)
                txn_date = month_start.replace(day=day)
                if self.start_date <= txn_date < self.end_date:
                    amount = self._sample_amount(TransactionCategory.SUBSCRIPTIONS, 1.0)
                    transactions.append(self._make_transaction(
                        persona.user_id, txn_date, -amount, TransactionCategory.SUBSCRIPTIONS
                    ))

            # Advance to next month
            if month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:
                current_date = current_date.replace(month=month + 1, day=1)

        return transactions

    def _generate_variable(
        self, persona: UserPersona, income_mult: float, target_count: int
    ) -> list[Transaction]:
        """Generate variable/sporadic transactions."""
        if target_count <= 0:
            return []

        transactions = []
        variable_categories = [
            TransactionCategory.GROCERIES,
            TransactionCategory.DINING,
            TransactionCategory.COFFEE,
            TransactionCategory.TRANSPORT,
            TransactionCategory.SHOPPING,
            TransactionCategory.HEALTHCARE,
            TransactionCategory.ENTERTAINMENT,
            TransactionCategory.TRANSFERS,
        ]

        # Category weights (persona-conditioned)
        weights = self._get_category_weights(persona)

        for _ in range(target_count):
            # Pick category (use index to avoid numpy string coercion)
            cat_idx = self.rng.choice(len(variable_categories), p=weights)
            cat = variable_categories[cat_idx]

            # Pick date with temporal bias
            txn_date = self._sample_date(cat)

            # Sample amount
            amount = self._sample_amount(cat, income_mult)

            # Apply seasonal multiplier
            seasonal = self.SEASONAL_MULTIPLIERS.get(txn_date.month, 1.0)
            amount *= seasonal

            # Determine sign
            _, _, _, _, is_inflow = self.CATEGORY_PARAMS[cat]
            if not is_inflow:
                amount = -amount

            transactions.append(self._make_transaction(
                persona.user_id, txn_date, amount, cat
            ))

        return transactions

    def _get_category_weights(self, persona: UserPersona) -> list[float]:
        """Get persona-conditioned category probabilities."""
        # Base weights
        weights = np.array([0.20, 0.18, 0.12, 0.15, 0.12, 0.03, 0.10, 0.10])

        # Adjust by persona
        if persona.digital_affinity > 0.7:
            weights[1] *= 1.3  # more dining (delivery)
            weights[4] *= 1.2  # more shopping (online)
        if persona.income_bracket in (IncomeBracket.HIGH, IncomeBracket.PREMIUM):
            weights[1] *= 1.2  # more dining
            weights[6] *= 1.3  # more entertainment
        if persona.spending_discipline > 0.7:
            weights[4] *= 0.7  # less shopping
            weights[6] *= 0.7  # less entertainment

        # Normalize
        weights = weights / weights.sum()
        return weights.tolist()

    def _sample_amount(self, category: TransactionCategory, multiplier: float) -> float:
        """Sample transaction amount from category distribution."""
        dist, mean, std, _, _ = self.CATEGORY_PARAMS[category]

        if dist == "normal":
            amount = self.rng.normal(mean * multiplier, std * np.sqrt(multiplier))
        else:  # lognormal
            # Shift mean in log-space by multiplier
            log_mult = np.log(max(0.1, multiplier))
            amount = self.rng.lognormal(mean + log_mult * 0.5, std)

        return max(0.01, round(abs(amount), 2))

    def _sample_date(self, category: TransactionCategory) -> datetime:
        """Sample a date with temporal bias for the category."""
        # Random day in range
        day_offset = self.rng.integers(0, self.total_days)
        date = self.start_date + timedelta(days=int(day_offset))

        # Weekday bias for certain categories
        weekday = date.weekday()  # 0=Mon, 6=Sun

        if category in (TransactionCategory.COFFEE, TransactionCategory.TRANSPORT):
            # Prefer weekdays - resample if weekend (70% chance)
            if weekday >= 5 and self.rng.random() < 0.7:
                date -= timedelta(days=weekday - 4)  # shift to Friday

        elif category in (TransactionCategory.DINING, TransactionCategory.ENTERTAINMENT):
            # Prefer weekends - shift toward Fri-Sun
            if weekday < 4 and self.rng.random() < 0.4:
                date += timedelta(days=5 - weekday)  # shift to Saturday

        elif category == TransactionCategory.SHOPPING:
            # Slight weekend bias
            if weekday < 5 and self.rng.random() < 0.3:
                date += timedelta(days=5 - weekday)

        # Add time-of-day
        hour = self._sample_hour(category)
        minute = self.rng.integers(0, 60)
        date = date.replace(hour=hour, minute=int(minute), second=0)

        # Clamp to valid range
        if date < self.start_date:
            date = self.start_date.replace(hour=hour, minute=int(minute))
        elif date >= self.end_date:
            date = self.end_date - timedelta(hours=1)

        return date

    def _sample_hour(self, category: TransactionCategory) -> int:
        """Sample hour of day based on category."""
        hour_ranges = {
            TransactionCategory.COFFEE: (6, 11),
            TransactionCategory.TRANSPORT: (6, 22),
            TransactionCategory.GROCERIES: (8, 21),
            TransactionCategory.DINING: (11, 23),
            TransactionCategory.SHOPPING: (9, 22),
            TransactionCategory.ENTERTAINMENT: (14, 23),
            TransactionCategory.HEALTHCARE: (7, 18),
            TransactionCategory.TRANSFERS: (8, 20),
        }
        low, high = hour_ranges.get(category, (8, 22))
        return int(self.rng.integers(low, high))

    def _make_transaction(
        self,
        user_id: str,
        timestamp: datetime,
        amount: float,
        category: TransactionCategory,
    ) -> Transaction:
        """Create a Transaction object."""
        _, _, _, txn_type, is_inflow = self.CATEGORY_PARAMS[category]

        # Pick merchant (use index to avoid numpy coercion)
        merchants = self.MERCHANTS[category]
        merchant_idx = self.rng.integers(0, len(merchants))
        merchant = merchants[merchant_idx]

        # Build description
        description = f"{merchant} - {category.value}"

        # Installments (only for credit, shopping/healthcare)
        installments = 1
        if txn_type == TransactionType.CREDIT and abs(amount) > 100:
            if self.rng.random() < 0.3:
                options = [2, 3, 4, 6, 10, 12]
                installments = options[int(self.rng.integers(0, len(options)))]

        # Status (99.5% approved)
        status = TransactionStatus.APPROVED
        if self.rng.random() < 0.005:
            statuses = [TransactionStatus.DENIED, TransactionStatus.REVERSED]
            status = statuses[int(self.rng.integers(0, 2))]

        return Transaction(
            user_id=user_id,
            timestamp=timestamp,
            amount=round(amount, 2),
            description=description,
            category=category,
            merchant_id=f"m_{merchant.lower()[:10]}_{self.rng.integers(100, 999)}",
            merchant_category_code=self.MCC_CODES[category],
            transaction_type=txn_type,
            installments=installments,
            status=status,
        )
