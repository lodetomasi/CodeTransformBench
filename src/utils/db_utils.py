"""
Database utilities for CodeTransformBench.
Provides connection management, session creation, and common queries.
"""

from contextlib import contextmanager
from typing import List, Optional, Dict, Any
from datetime import date

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import config
from database.models import Base, Function, Transformation, CostTracking


# Create engine (lazy initialization)
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create database engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            config.DATABASE_URL,
            poolclass=NullPool,  # Don't pool connections (simpler for research use)
            echo=False,  # Set to True for SQL debugging
        )
    return _engine


def get_session_maker():
    """Get or create session maker."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine()
        )
    return _SessionLocal


@contextmanager
def get_db_session():
    """
    Context manager for database sessions.

    Usage:
        with get_db_session() as session:
            session.query(Function).all()
    """
    SessionLocal = get_session_maker()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def create_all_tables():
    """Create all tables in the database (if they don't exist)."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


# ============================================================================
# FUNCTION QUERIES
# ============================================================================

def get_function_by_id(function_id: str) -> Optional[Function]:
    """Get function by ID."""
    with get_db_session() as session:
        return session.query(Function).filter(Function.id == function_id).first()


def get_all_functions(validated_only: bool = True) -> List[Function]:
    """Get all functions, optionally filtered by validation status."""
    with get_db_session() as session:
        query = session.query(Function)
        if validated_only:
            query = query.filter(Function.validated == True)
        return query.all()


def get_functions_by_language(language: str, validated_only: bool = True) -> List[Function]:
    """Get functions for a specific language."""
    with get_db_session() as session:
        query = session.query(Function).filter(Function.language == language)
        if validated_only:
            query = query.filter(Function.validated == True)
        return query.all()


def get_functions_by_complexity_tier(tier: str, validated_only: bool = True) -> List[Function]:
    """
    Get functions by complexity tier.

    Args:
        tier: 'simple' (CC≤10), 'medium' (CC 11-30), or 'complex' (CC≥31)
    """
    with get_db_session() as session:
        query = session.query(Function)

        if tier == 'simple':
            query = query.filter(Function.cyclomatic_complexity <= 10)
        elif tier == 'medium':
            query = query.filter(
                Function.cyclomatic_complexity > 10,
                Function.cyclomatic_complexity <= 30
            )
        elif tier == 'complex':
            query = query.filter(Function.cyclomatic_complexity > 30)
        else:
            raise ValueError(f"Invalid tier: {tier}")

        if validated_only:
            query = query.filter(Function.validated == True)

        return query.all()


def count_functions_by_language() -> Dict[str, int]:
    """Get function count grouped by language."""
    with get_db_session() as session:
        results = session.query(
            Function.language,
            func.count(Function.id)
        ).group_by(Function.language).all()

        return {lang: count for lang, count in results}


# ============================================================================
# TRANSFORMATION QUERIES
# ============================================================================

def get_transformation_by_id(transformation_id: int) -> Optional[Transformation]:
    """Get transformation by ID."""
    with get_db_session() as session:
        return session.query(Transformation).filter(Transformation.id == transformation_id).first()


def get_transformations_for_function(function_id: str) -> List[Transformation]:
    """Get all transformations for a specific function."""
    with get_db_session() as session:
        return session.query(Transformation).filter(
            Transformation.function_id == function_id
        ).all()


def check_transformation_exists(
    function_id: str,
    model: str,
    task: str,
    strategy: str
) -> bool:
    """Check if a transformation already exists (for caching)."""
    with get_db_session() as session:
        result = session.query(Transformation).filter(
            Transformation.function_id == function_id,
            Transformation.model == model,
            Transformation.task == task,
            Transformation.strategy == strategy
        ).first()
        return result is not None


def get_cached_transformation(
    function_id: str,
    model: str,
    task: str,
    strategy: str
) -> Optional[Transformation]:
    """Get cached transformation if it exists."""
    with get_db_session() as session:
        return session.query(Transformation).filter(
            Transformation.function_id == function_id,
            Transformation.model == model,
            Transformation.task == task,
            Transformation.strategy == strategy
        ).first()


def get_leaderboard(
    task: Optional[str] = None,
    strategy: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get leaderboard data (aggregated transformation stats).

    Returns results from the leaderboard VIEW.
    """
    with get_db_session() as session:
        query = "SELECT * FROM leaderboard WHERE 1=1"
        params = {}

        if task:
            query += " AND task = :task"
            params['task'] = task

        if strategy:
            query += " AND strategy = :strategy"
            params['strategy'] = strategy

        query += " ORDER BY mean_se DESC LIMIT :limit"
        params['limit'] = limit

        result = session.execute(query, params)
        columns = result.keys()
        return [dict(zip(columns, row)) for row in result.fetchall()]


# ============================================================================
# COST TRACKING QUERIES
# ============================================================================

def get_total_cost() -> float:
    """Get total cost across all experiments."""
    with get_db_session() as session:
        result = session.query(func.sum(CostTracking.total_cost_usd)).scalar()
        return float(result) if result else 0.0


def get_total_cost_by_tier(tier: str) -> float:
    """Get total cost for a specific tier."""
    from src.config import get_models_by_tier

    tier_models = get_models_by_tier(tier)
    model_names = [m['id'] for m in tier_models]

    with get_db_session() as session:
        result = session.query(func.sum(CostTracking.total_cost_usd)).filter(
            CostTracking.model.in_(model_names)
        ).scalar()
        return float(result) if result else 0.0


def get_cost_by_model(model: str) -> float:
    """Get total cost for a specific model."""
    with get_db_session() as session:
        result = session.query(func.sum(CostTracking.total_cost_usd)).filter(
            CostTracking.model == model
        ).scalar()
        return float(result) if result else 0.0


def get_daily_cost(target_date: Optional[date] = None) -> float:
    """Get total cost for a specific date (defaults to today)."""
    if target_date is None:
        target_date = date.today()

    with get_db_session() as session:
        result = session.query(func.sum(CostTracking.total_cost_usd)).filter(
            CostTracking.date == target_date
        ).scalar()
        return float(result) if result else 0.0


def get_remaining_budget(tier: Optional[str] = None) -> float:
    """Get remaining budget (total or for a specific tier)."""
    if tier:
        if tier == 'tier1':
            budget = config.BUDGET_TIER1_USD
        elif tier == 'tier2':
            budget = config.BUDGET_TIER2_USD
        elif tier == 'tier3':
            budget = config.BUDGET_TIER3_USD
        else:
            raise ValueError(f"Invalid tier: {tier}")

        spent = get_total_cost_by_tier(tier)
    else:
        budget = config.BUDGET_TOTAL_USD
        spent = get_total_cost()

    return budget - spent


def is_budget_exhausted(tier: Optional[str] = None, threshold: float = 0.95) -> bool:
    """
    Check if budget is exhausted (>95% spent by default).

    Args:
        tier: Check specific tier or total budget
        threshold: Percentage threshold (0.95 = 95%)
    """
    if tier:
        if tier == 'tier1':
            budget = config.BUDGET_TIER1_USD
        elif tier == 'tier2':
            budget = config.BUDGET_TIER2_USD
        elif tier == 'tier3':
            budget = config.BUDGET_TIER3_USD
        else:
            raise ValueError(f"Invalid tier: {tier}")

        spent = get_total_cost_by_tier(tier)
    else:
        budget = config.BUDGET_TOTAL_USD
        spent = get_total_cost()

    return (spent / budget) >= threshold


# ============================================================================
# STATISTICS
# ============================================================================

def get_dataset_stats() -> Dict[str, Any]:
    """Get overall dataset statistics."""
    with get_db_session() as session:
        total_functions = session.query(func.count(Function.id)).scalar()
        validated_functions = session.query(func.count(Function.id)).filter(
            Function.validated == True
        ).scalar()
        total_transformations = session.query(func.count(Transformation.id)).scalar()
        successful_transformations = session.query(func.count(Transformation.id)).filter(
            Transformation.preservation == 1
        ).scalar()

        return {
            'total_functions': total_functions,
            'validated_functions': validated_functions,
            'total_transformations': total_transformations,
            'successful_transformations': successful_transformations,
            'success_rate': (
                successful_transformations / total_transformations * 100
                if total_transformations > 0 else 0
            ),
            'total_cost_usd': get_total_cost(),
            'remaining_budget_usd': get_remaining_budget()
        }


if __name__ == '__main__':
    # Test database connection
    print("Testing database utilities...")

    try:
        engine = get_engine()
        print(f"✓ Connected to database: {config.DATABASE_URL}")

        stats = get_dataset_stats()
        print(f"\nDataset statistics:")
        print(f"  Functions: {stats['total_functions']} ({stats['validated_functions']} validated)")
        print(f"  Transformations: {stats['total_transformations']} ({stats['successful_transformations']} successful)")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
        print(f"  Total cost: ${stats['total_cost_usd']:.2f}")
        print(f"  Remaining budget: ${stats['remaining_budget_usd']:.2f}")

    except Exception as e:
        print(f"✗ Error: {e}")
