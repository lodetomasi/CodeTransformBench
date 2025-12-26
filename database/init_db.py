#!/usr/bin/env python3
"""
Database initialization script for CodeTransformBench.
Creates tables, indexes, and triggers from schema.sql.
Verifies setup is correct.
"""

import os
import sys
from pathlib import Path
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()


def get_db_connection():
    """Create PostgreSQL connection from DATABASE_URL."""
    database_url = os.getenv('DATABASE_URL', 'postgresql://localhost/codetransform')

    # Parse database URL
    if database_url.startswith('postgresql://'):
        # Extract components
        parts = database_url.replace('postgresql://', '').split('/')
        host_port = parts[0].split('@')[-1] if '@' in parts[0] else parts[0]
        host = host_port.split(':')[0]
        dbname = parts[1] if len(parts) > 1 else 'codetransform'

        return psycopg2.connect(
            host=host,
            database=dbname,
            user=os.getenv('DB_USER', os.environ.get('USER', 'postgres')),
            password=os.getenv('DB_PASSWORD', '')
        )
    else:
        raise ValueError(f"Invalid DATABASE_URL format: {database_url}")


def execute_schema_file(conn, schema_path):
    """Execute SQL schema file."""
    with open(schema_path, 'r') as f:
        schema_sql = f.read()

    with conn.cursor() as cur:
        cur.execute(schema_sql)
    conn.commit()
    print(f"✓ Executed schema from {schema_path}")


def verify_tables(conn):
    """Verify all tables were created."""
    expected_tables = {'functions', 'transformations', 'cost_tracking'}

    with conn.cursor() as cur:
        cur.execute("""
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
        """)
        tables = {row[0] for row in cur.fetchall()}

    missing = expected_tables - tables
    if missing:
        print(f"✗ Missing tables: {missing}")
        return False

    print(f"✓ All tables created: {expected_tables}")
    return True


def verify_indexes(conn):
    """Verify key indexes were created."""
    expected_indexes = {
        'idx_functions_complexity',
        'idx_functions_language',
        'idx_transformations_model_task',
    }

    with conn.cursor() as cur:
        cur.execute("""
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
        """)
        indexes = {row[0] for row in cur.fetchall()}

    missing = expected_indexes - indexes
    if missing:
        print(f"⚠ Some indexes missing: {missing}")

    print(f"✓ Found {len(indexes)} indexes")
    return True


def verify_trigger(conn):
    """Verify cost tracking trigger was created."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT tgname
            FROM pg_trigger
            WHERE tgname = 'trigger_update_cost_tracking'
        """)
        trigger = cur.fetchone()

    if not trigger:
        print("✗ Trigger 'trigger_update_cost_tracking' not found")
        return False

    print("✓ Trigger 'trigger_update_cost_tracking' created")
    return True


def test_trigger(conn):
    """Test that cost tracking trigger works."""
    with conn.cursor() as cur:
        # Insert a test function
        cur.execute("""
            INSERT INTO functions (id, language, code, code_hash,
                                   cyclomatic_complexity, lines_of_code, source)
            VALUES ('test_001', 'python', 'def test(): pass',
                    '0000000000000000000000000000000000000000000000000000000000000000',
                    1, 1, 'test')
            ON CONFLICT (code_hash) DO NOTHING
        """)

        # Insert a transformation (should trigger cost_tracking update)
        cur.execute("""
            INSERT INTO transformations
                (function_id, model, task, strategy, preservation, cost_usd, latency_ms)
            VALUES ('test_001', 'test-model', 'obfuscate', 'zero_shot', 1, 0.001, 100)
            ON CONFLICT (function_id, model, task, strategy) DO NOTHING
        """)

        # Check if cost_tracking was updated
        cur.execute("""
            SELECT total_calls, total_cost_usd
            FROM cost_tracking
            WHERE model = 'test-model' AND date = CURRENT_DATE
        """)
        result = cur.fetchone()

        if result and result[0] > 0:
            print(f"✓ Trigger working: {result[0]} calls, ${result[1]:.4f}")

            # Cleanup test data
            cur.execute("DELETE FROM transformations WHERE model = 'test-model'")
            cur.execute("DELETE FROM cost_tracking WHERE model = 'test-model'")
            cur.execute("DELETE FROM functions WHERE id = 'test_001'")
            conn.commit()
            return True
        else:
            print("✗ Trigger test failed: cost_tracking not updated")
            conn.rollback()
            return False


def verify_view(conn):
    """Verify leaderboard view was created."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT viewname
            FROM pg_views
            WHERE schemaname = 'public' AND viewname = 'leaderboard'
        """)
        view = cur.fetchone()

    if not view:
        print("✗ View 'leaderboard' not found")
        return False

    print("✓ View 'leaderboard' created")
    return True


def main():
    """Main initialization function."""
    print("=" * 60)
    print("CodeTransformBench - Database Initialization")
    print("=" * 60)

    # Get schema path
    schema_path = Path(__file__).parent / 'schema.sql'
    if not schema_path.exists():
        print(f"✗ Schema file not found: {schema_path}")
        sys.exit(1)

    try:
        # Connect to database
        print("\n1. Connecting to database...")
        conn = get_db_connection()
        print(f"✓ Connected to database")

        # Execute schema
        print("\n2. Creating tables, indexes, and triggers...")
        execute_schema_file(conn, schema_path)

        # Verify setup
        print("\n3. Verifying database setup...")
        all_ok = True
        all_ok &= verify_tables(conn)
        all_ok &= verify_indexes(conn)
        all_ok &= verify_trigger(conn)
        all_ok &= verify_view(conn)

        # Test trigger
        print("\n4. Testing cost tracking trigger...")
        all_ok &= test_trigger(conn)

        # Final status
        print("\n" + "=" * 60)
        if all_ok:
            print("✓ Database initialization SUCCESSFUL")
            print("=" * 60)
            print("\nNext steps:")
            print("  1. Copy .env.example to .env")
            print("  2. Add your OpenRouter API key to .env")
            print("  3. Install Python dependencies: pip install -r requirements.txt")
            print("  4. Start collecting data: python src/collectors/rosetta_scraper.py")
        else:
            print("✗ Database initialization completed with WARNINGS")
            print("=" * 60)

        conn.close()

    except psycopg2.Error as e:
        print(f"\n✗ Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
