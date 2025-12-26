"""
Configuration loader for CodeTransformBench.
Loads environment variables and YAML config files.
"""

import os
from pathlib import Path
from typing import Dict, List, Any
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / 'config'
DATA_DIR = PROJECT_ROOT / 'data'
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'
RESULTS_DIR = PROJECT_ROOT / 'results'


class Config:
    """Central configuration class."""

    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/codetransform')

    # OpenRouter API
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'

    # Budget (in USD)
    BUDGET_TOTAL_USD = float(os.getenv('BUDGET_TOTAL_USD', 2000))
    BUDGET_TIER1_PCT = float(os.getenv('BUDGET_TIER1_PCT', 60))
    BUDGET_TIER2_PCT = float(os.getenv('BUDGET_TIER2_PCT', 30))
    BUDGET_TIER3_PCT = float(os.getenv('BUDGET_TIER3_PCT', 10))

    @property
    def BUDGET_TIER1_USD(self):
        return self.BUDGET_TOTAL_USD * (self.BUDGET_TIER1_PCT / 100)

    @property
    def BUDGET_TIER2_USD(self):
        return self.BUDGET_TOTAL_USD * (self.BUDGET_TIER2_PCT / 100)

    @property
    def BUDGET_TIER3_USD(self):
        return self.BUDGET_TOTAL_USD * (self.BUDGET_TIER3_PCT / 100)

    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE = int(os.getenv('MAX_REQUESTS_PER_MINUTE', 60))
    MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', 10))

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'codetransform.log')

    # Paths
    DATA_RAW_DIR = DATA_DIR / 'raw'
    DATA_PROCESSED_DIR = DATA_DIR / 'processed'
    DATA_TEST_SUITES_DIR = DATA_DIR / 'test_suites'
    ROSETTA_HTML_DIR = DATA_RAW_DIR / 'rosetta_html'
    ALGORITHMS_REPOS_DIR = DATA_RAW_DIR / 'algorithms_repos'

    PROMPTS_DIR = EXPERIMENTS_DIR / 'prompts'
    PROMPTS_EXAMPLES_DIR = PROMPTS_DIR / 'examples'
    EXPERIMENTS_CONFIGS_DIR = EXPERIMENTS_DIR / 'configs'

    RESULTS_RAW_DIR = RESULTS_DIR / 'raw_api_responses'
    RESULTS_ANALYSIS_DIR = RESULTS_DIR / 'analysis'

    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        errors = []

        if not cls.OPENROUTER_API_KEY:
            errors.append("OPENROUTER_API_KEY not set in environment")

        if cls.BUDGET_TOTAL_USD <= 0:
            errors.append("BUDGET_TOTAL_USD must be positive")

        if cls.BUDGET_TIER1_PCT + cls.BUDGET_TIER2_PCT + cls.BUDGET_TIER3_PCT != 100:
            errors.append("Tier budget percentages must sum to 100")

        if errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

        return True


def load_models_config() -> Dict[str, List[Dict[str, Any]]]:
    """Load model configurations from config/models.yaml."""
    config_path = CONFIG_DIR / 'models.yaml'

    if not config_path.exists():
        raise FileNotFoundError(f"Models config not found: {config_path}")

    with open(config_path, 'r') as f:
        models_config = yaml.safe_load(f)

    return models_config


def load_experiment_config(config_name: str) -> Dict[str, Any]:
    """Load experiment configuration from config/{config_name}.yaml."""
    config_path = CONFIG_DIR / f'{config_name}.yaml'

    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    with open(config_path, 'r') as f:
        experiment_config = yaml.safe_load(f)

    return experiment_config


def load_prompt_template(template_name: str) -> str:
    """Load prompt template from experiments/prompts/{template_name}.txt."""
    template_path = EXPERIMENTS_DIR / 'prompts' / f'{template_name}.txt'

    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    with open(template_path, 'r') as f:
        template = f.read()

    return template


def get_model_by_id(model_id: str) -> Dict[str, Any]:
    """Get model configuration by OpenRouter model ID."""
    models_config = load_models_config()

    for tier in ['tier1', 'tier2', 'tier3']:
        for model in models_config.get(tier, []):
            if model['id'] == model_id:
                model['tier'] = tier
                return model

    raise ValueError(f"Model not found: {model_id}")


def get_models_by_tier(tier: str) -> List[Dict[str, Any]]:
    """Get all models for a specific tier."""
    models_config = load_models_config()

    if tier not in models_config:
        raise ValueError(f"Invalid tier: {tier}. Must be tier1, tier2, or tier3")

    models = models_config[tier]
    for model in models:
        model['tier'] = tier

    return models


def get_all_models() -> List[Dict[str, Any]]:
    """Get all models across all tiers."""
    models_config = load_models_config()
    all_models = []

    for tier in ['tier1', 'tier2', 'tier3']:
        models = models_config.get(tier, [])
        for model in models:
            model['tier'] = tier
            all_models.append(model)

    return all_models


# Create instance for easy import
config = Config()


if __name__ == '__main__':
    # Test configuration
    print("Testing configuration...")

    try:
        config.validate()
        print("✓ Configuration valid")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")

    print(f"\nDatabase URL: {config.DATABASE_URL}")
    print(f"Budget: ${config.BUDGET_TOTAL_USD}")
    print(f"  Tier 1: ${config.BUDGET_TIER1_USD}")
    print(f"  Tier 2: ${config.BUDGET_TIER2_USD}")
    print(f"  Tier 3: ${config.BUDGET_TIER3_USD}")

    if config.OPENROUTER_API_KEY:
        print(f"\nAPI Key: {config.OPENROUTER_API_KEY[:10]}...")
    else:
        print("\n⚠ API Key not set")

    print(f"\nRate limits: {config.MAX_REQUESTS_PER_MINUTE} req/min, {config.MAX_CONCURRENT_REQUESTS} concurrent")
