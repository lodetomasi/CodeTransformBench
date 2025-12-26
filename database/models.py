"""
SQLAlchemy ORM models for CodeTransformBench database.
Maps to PostgreSQL schema defined in schema.sql.
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text,
    DateTime, Date, ForeignKey, CheckConstraint, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Function(Base):
    """Source code corpus table."""
    __tablename__ = 'functions'

    id = Column(String(100), primary_key=True)
    language = Column(String(20), nullable=False)
    code = Column(Text, nullable=False)
    code_hash = Column(String(64), unique=True, nullable=False)
    cyclomatic_complexity = Column(Integer, nullable=False)
    halstead_volume = Column(Float)
    lines_of_code = Column(Integer, nullable=False)
    domain = Column(String(50))
    source = Column(String(50), nullable=False)
    task_name = Column(String(200))
    test_suite_path = Column(String(255))
    validated = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to transformations
    transformations = relationship(
        "Transformation",
        back_populates="function",
        cascade="all, delete-orphan"
    )

    # Check constraints
    __table_args__ = (
        CheckConstraint(
            language.in_(['python', 'java', 'javascript', 'cpp']),
            name='check_language'
        ),
        CheckConstraint(
            'cyclomatic_complexity >= 1',
            name='check_cc_positive'
        ),
        CheckConstraint(
            'lines_of_code > 0',
            name='check_loc_positive'
        ),
    )

    def __repr__(self):
        return f"<Function {self.id} ({self.language}, CC={self.cyclomatic_complexity})>"

    @property
    def complexity_tier(self):
        """Return complexity classification."""
        if self.cyclomatic_complexity <= 10:
            return 'simple'
        elif self.cyclomatic_complexity <= 30:
            return 'medium'
        else:
            return 'complex'


class Transformation(Base):
    """Transformation experiment results table."""
    __tablename__ = 'transformations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    function_id = Column(String(100), ForeignKey('functions.id', ondelete='CASCADE'), nullable=False)
    model = Column(String(100), nullable=False)
    task = Column(String(50), nullable=False)
    strategy = Column(String(50), nullable=False)
    temperature = Column(Float, default=0.2)
    transformed_code = Column(Text)

    # Semantic Elasticity components
    delta_cc = Column(Float)
    preservation = Column(Integer)
    diversity = Column(Float)
    effort = Column(Float)
    se_score = Column(Float)

    # API metadata
    cost_usd = Column(Float)
    latency_ms = Column(Integer)
    tokens_input = Column(Integer)
    tokens_output = Column(Integer)
    error_type = Column(String(50))

    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationship to function
    function = relationship("Function", back_populates="transformations")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            task.in_(['obfuscate', 'deobfuscate', 'refactor']),
            name='check_task'
        ),
        CheckConstraint(
            strategy.in_(['zero_shot', 'few_shot_k3', 'few_shot_k5', 'chain_of_thought', 'self_reflection']),
            name='check_strategy'
        ),
        CheckConstraint(
            'preservation IN (0, 1)',
            name='check_preservation'
        ),
        UniqueConstraint(
            'function_id', 'model', 'task', 'strategy',
            name='unique_experiment'
        ),
    )

    def __repr__(self):
        return (
            f"<Transformation {self.id}: {self.model} {self.task} "
            f"(SE={self.se_score:.2f if self.se_score else 'N/A'})>"
        )

    @property
    def success(self):
        """Returns True if transformation preserved functionality."""
        return self.preservation == 1

    @property
    def total_tokens(self):
        """Returns total tokens used (input + output)."""
        return (self.tokens_input or 0) + (self.tokens_output or 0)


class CostTracking(Base):
    """Daily cost tracking by model."""
    __tablename__ = 'cost_tracking'

    date = Column(Date, primary_key=True, nullable=False)
    model = Column(String(100), primary_key=True, nullable=False)
    total_calls = Column(Integer, default=0)
    total_cost_usd = Column(Float, default=0.0)
    avg_latency_ms = Column(Float)
    successful_calls = Column(Integer, default=0)
    failed_calls = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return (
            f"<CostTracking {self.date} {self.model}: "
            f"${self.total_cost_usd:.2f} ({self.total_calls} calls)>"
        )

    @property
    def success_rate(self):
        """Returns success rate as percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100

    @property
    def cost_per_call(self):
        """Returns average cost per call."""
        if self.total_calls == 0:
            return 0.0
        return self.total_cost_usd / self.total_calls
