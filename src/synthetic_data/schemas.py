"""Data schemas for synthetic financial transaction generation.

Defines Pydantic models for:
- UserPersona: Demographics and financial profile
- Transaction: Individual financial transaction
- TabularFeatures: Computed user-level features (291 dimensions)
- DatasetConfig: Generation parameters
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class IncomeBracket(str, Enum):
    LOW = "low"
    MIDDLE = "middle"
    HIGH = "high"
    PREMIUM = "premium"


class AgeGroup(str, Enum):
    GEN_Z = "gen_z"  # 18-28
    MILLENNIAL = "millennial"  # 29-44
    GEN_X = "gen_x"  # 45-60
    BOOMER = "boomer"  # 61+


class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class TransactionType(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"
    PIX = "pix"
    TRANSFER = "transfer"
    BOLETO = "boleto"


class TransactionCategory(str, Enum):
    SALARY = "salary"
    RENT = "rent"
    GROCERIES = "groceries"
    DINING = "dining"
    COFFEE = "coffee"
    TRANSPORT = "transport"
    SHOPPING = "shopping"
    UTILITIES = "utilities"
    SUBSCRIPTIONS = "subscriptions"
    HEALTHCARE = "healthcare"
    ENTERTAINMENT = "entertainment"
    TRANSFERS = "transfers"


class TransactionStatus(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    PENDING = "pending"
    REVERSED = "reversed"


class UserPersona(BaseModel):
    """User demographic and financial profile for persona-conditioned generation."""

    user_id: str
    income_bracket: IncomeBracket
    age_group: AgeGroup
    risk_profile: RiskProfile
    monthly_income: float = Field(ge=500, le=50000)
    credit_limit: float = Field(ge=0, le=100000)
    credit_score: int = Field(ge=300, le=850)
    account_age_days: int = Field(ge=30, le=3650)
    is_primary_bank: bool = True
    region: str = "southeast"
    has_credit_card: bool = False  # label: will they activate?

    # Spending personality (affects generation)
    spending_discipline: float = Field(ge=0.0, le=1.0, default=0.5)
    digital_affinity: float = Field(ge=0.0, le=1.0, default=0.7)


class Transaction(BaseModel):
    """Individual financial transaction."""

    user_id: str
    timestamp: datetime
    amount: float  # positive=inflow, negative=outflow
    description: str
    category: TransactionCategory
    merchant_id: str
    merchant_category_code: str  # MCC code
    transaction_type: TransactionType
    installments: int = Field(ge=1, le=24, default=1)
    status: TransactionStatus = TransactionStatus.APPROVED


class TabularFeatures(BaseModel):
    """User-level aggregated features (291 dimensions).

    Grouped into:
    - Transaction aggregates (~100 features)
    - Temporal patterns (~50 features)
    - Merchant diversity (~30 features)
    - Financial health (~40 features)
    - Bureau/external (~30 features)
    - Behavioral (~30 features)
    - Demographics (~11 features)
    """

    user_id: str

    # Transaction aggregates (sample - full list computed dynamically)
    total_spend_30d: float = 0.0
    total_spend_90d: float = 0.0
    total_spend_180d: float = 0.0
    total_income_30d: float = 0.0
    avg_transaction_amount: float = 0.0
    median_transaction_amount: float = 0.0
    max_transaction_amount: float = 0.0
    transaction_count_30d: int = 0
    transaction_count_90d: int = 0

    # Category shares
    groceries_share: float = 0.0
    dining_share: float = 0.0
    transport_share: float = 0.0
    entertainment_share: float = 0.0

    # Temporal
    weekday_spend_ratio: float = 0.0
    weekend_spike_factor: float = 0.0
    morning_activity_ratio: float = 0.0
    evening_activity_ratio: float = 0.0
    days_since_last_txn: int = 0
    txn_frequency_std: float = 0.0

    # Merchant diversity
    unique_merchants_30d: int = 0
    unique_merchants_90d: int = 0
    merchant_concentration: float = 0.0  # HHI
    top_merchant_share: float = 0.0

    # Financial health
    credit_utilization: float = 0.0
    savings_rate: float = 0.0
    debt_to_income: float = 0.0
    overdraft_count_90d: int = 0
    balance_volatility: float = 0.0

    # Bureau (simulated)
    credit_score: int = 650
    inquiry_count_6m: int = 0
    delinquency_flag: bool = False
    accounts_open: int = 1
    oldest_account_months: int = 12

    # Target label
    activated_credit_card: Optional[bool] = None


class DatasetConfig(BaseModel):
    """Configuration for synthetic data generation."""

    # Scale
    num_users: int = 1000
    min_transactions_per_user: int = 50
    max_transactions_per_user: int = 2000
    target_total_transactions: Optional[int] = None  # e.g., 100_000_000

    # Time range
    start_date: str = "2022-01-01"
    end_date: str = "2024-06-30"  # 2.5 years

    # Label
    label_window_days: int = 180  # 6 months forward
    positive_rate: float = 0.15  # 15% activate credit card

    # Tabular
    num_tabular_features: int = 291

    # Output
    output_dir: str = "data/raw"
    format: str = "parquet"  # parquet or csv
    seed: int = 42

    # Splits (temporal)
    train_end_date: str = "2023-12-31"  # first 2 years
    val_end_date: str = "2024-03-31"  # next 3 months
    # test: 2024-04-01 to 2024-06-30 (last 3 months)
