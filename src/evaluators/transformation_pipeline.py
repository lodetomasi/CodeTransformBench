#!/usr/bin/env python3
"""
Transformation Pipeline for CodeTransformBench

Executes code transformations using LLMs with:
- Checkpoint/resume capability
- Budget monitoring
- Error recovery
- Progress tracking
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

from src.api.openrouter_client import OpenRouterClient
from src.utils.db_utils import get_db_session
from src.utils.logger import logger
from src.collectors.metrics_calculator import calculate_metrics
from database.models import Function, Transformation
from sqlalchemy import and_

class TransformationPipeline:
    """
    Orchestrates code transformations with checkpoint and budget management.
    """

    def __init__(self, budget_limit_usd: float = 2000.0):
        """
        Initialize transformation pipeline.

        Args:
            budget_limit_usd: Maximum budget for all transformations (default: $2000)
        """
        self.client = OpenRouterClient()
        self.budget_limit = budget_limit_usd
        self.prompt_dir = Path(__file__).parent.parent.parent / "experiments" / "prompts"

        logger.info(f"Initialized transformation pipeline with ${budget_limit_usd:.2f} budget")

    def load_prompt_template(self, task: str, strategy: str = "zero_shot") -> str:
        """
        Load prompt template for transformation task.

        Args:
            task: Transformation task (obfuscate, deobfuscate, refactor)
            strategy: Prompting strategy (zero_shot, few_shot_k3, etc.)

        Returns:
            Prompt template string
        """
        template_path = self.prompt_dir / f"{task}_{strategy}.txt"

        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")

        return template_path.read_text()

    def render_prompt(self, template: str, code: str, language: str) -> str:
        """
        Render prompt template with code.

        Args:
            template: Prompt template string
            code: Source code to transform
            language: Programming language

        Returns:
            Rendered prompt
        """
        return template.replace("{code}", code).replace("{language}", language)

    def extract_code_from_response(self, response: str, language: str) -> Optional[str]:
        """
        Extract code from LLM response (same as test_generator).

        Args:
            response: LLM response text
            language: Programming language

        Returns:
            Extracted code or None
        """
        # Try markdown code blocks first
        patterns = [
            r'```(?:' + language + r')?\s*\n(.*?)```',
            r'```\s*\n(.*?)```',
            r'```(?:' + language + r')?\s+(.*?)```',
            r'```\s+(.*?)```',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                code = max(matches, key=len).strip()
                if code:
                    return code

        # Language-specific fallbacks
        code_indicators = {
            'python': ['def ', 'import ', 'class ', 'from '],
            'javascript': ['function ', 'const ', 'let ', 'var '],
            'java': ['public ', 'class ', 'import ', 'private '],
            'cpp': ['#include', 'void ', 'int ', 'class ']
        }

        if language in code_indicators:
            if any(indicator in response for indicator in code_indicators[language]):
                cleaned = response.strip()
                cleaned = re.sub(r'^(Here is|Here\'s|This is).*?:\s*', '', cleaned, flags=re.IGNORECASE)
                if cleaned:
                    return cleaned

        logger.warning(f"Could not extract code from response for {language}")
        return None

    async def transform_code(
        self,
        function: Dict[str, Any],
        model: str,
        task: str,
        strategy: str = "zero_shot",
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Transform a single function using specified model and task.

        Args:
            function: Function dict with keys: id, code, language
            model: Model ID (e.g., 'anthropic/claude-3.5-sonnet')
            task: Task name (obfuscate, deobfuscate, refactor)
            strategy: Prompting strategy
            max_retries: Maximum retry attempts

        Returns:
            Transformation result dict or None if failed
        """
        func_id = function['id']
        language = function['language']
        code = function['code']

        logger.info(f"Transforming {func_id} with {model} (task: {task}, strategy: {strategy})")

        # Load and render prompt
        try:
            template = self.load_prompt_template(task, strategy)
            prompt = self.render_prompt(template, code, language)
        except Exception as e:
            logger.error(f"Failed to load prompt template: {e}")
            return None

        # Try transformation with retries
        for attempt in range(1, max_retries + 1):
            try:
                start_time = time.time()

                # Call LLM
                result = await self.client.generate(
                    model=model,
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=4000
                )

                latency_ms = result['latency_ms']

                # Extract code
                transformed_code = self.extract_code_from_response(result['content'], language)

                if not transformed_code:
                    logger.warning(f"{func_id}: Failed to extract code (attempt {attempt}/{max_retries})")
                    continue

                # Calculate metrics for transformed code
                try:
                    metrics = calculate_metrics(transformed_code, language)
                except Exception as e:
                    logger.warning(f"{func_id}: Invalid transformed code - {e} (attempt {attempt}/{max_retries})")
                    continue

                # Success!
                logger.success(f"Transformed {func_id} with {model} (cost: ${result['cost_usd']:.4f})")

                return {
                    'function_id': func_id,
                    'model': model,
                    'task': task,
                    'strategy': strategy,
                    'transformed_code': transformed_code,
                    'cyclomatic_complexity': metrics['cyclomatic_complexity'],
                    'halstead_volume': metrics['halstead_volume'],
                    'cost_usd': result['cost'],
                    'latency_ms': latency_ms,
                    'tokens_input': result['tokens_input'],
                    'tokens_output': result['tokens_output']
                }

            except Exception as e:
                logger.error(f"{func_id}: Error during transformation attempt {attempt}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        logger.error(f"Failed to transform {func_id} after {max_retries} attempts")
        return None

    def check_transformation_exists(
        self,
        function_id: str,
        model: str,
        task: str,
        strategy: str
    ) -> bool:
        """
        Check if transformation already exists in database (for resume capability).

        Args:
            function_id: Function ID
            model: Model ID
            task: Task name
            strategy: Strategy name

        Returns:
            True if transformation exists
        """
        with get_db_session() as session:
            exists = session.query(Transformation).filter(
                and_(
                    Transformation.function_id == function_id,
                    Transformation.model == model,
                    Transformation.task == task,
                    Transformation.strategy == strategy
                )
            ).first() is not None

            return exists

    async def run_transformations(
        self,
        functions: Optional[List[Dict[str, Any]]] = None,
        models: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
        max_concurrent: int = 5,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Run transformations for specified combinations.

        Args:
            functions: List of function dicts (if None, loads all from DB)
            models: List of model IDs (if None, uses all configured models)
            tasks: List of tasks (if None, uses all: obfuscate, deobfuscate, refactor)
            strategies: List of strategies (if None, uses zero_shot only)
            max_concurrent: Maximum concurrent API calls
            resume: If True, skip already-completed transformations

        Returns:
            Statistics dict
        """
        # Default values
        if tasks is None:
            tasks = ['obfuscate']  # Start with just obfuscate for testing

        if strategies is None:
            strategies = ['zero_shot']

        # Load functions if not provided
        if functions is None:
            with get_db_session() as session:
                db_functions = session.query(Function).all()
                functions = [
                    {
                        'id': f.id,
                        'language': f.language,
                        'code': f.code,
                        'cyclomatic_complexity': f.cyclomatic_complexity
                    }
                    for f in db_functions
                ]

        # Load models if not provided
        if models is None:
            from src.config import get_models_by_tier
            all_models = []
            for tier in ['tier1', 'tier2', 'tier3']:
                all_models.extend([m['id'] for m in get_models_by_tier(tier)])
            models = all_models

        logger.info(f"Running transformations: {len(functions)} functions × {len(models)} models × {len(tasks)} tasks × {len(strategies)} strategies")
        logger.info(f"Total combinations: {len(functions) * len(models) * len(tasks) * len(strategies)}")
        logger.info(f"Budget limit: ${self.budget_limit:.2f}")
        logger.info(f"Resume mode: {'ON' if resume else 'OFF'}")

        # Build work queue
        work_queue = []
        skipped = 0

        for func in functions:
            for model in models:
                for task in tasks:
                    for strategy in strategies:
                        # Check if already exists (resume capability)
                        if resume and self.check_transformation_exists(func['id'], model, task, strategy):
                            skipped += 1
                            continue

                        work_queue.append((func, model, task, strategy))

        logger.info(f"Work queue: {len(work_queue)} transformations (skipped {skipped} existing)")

        if len(work_queue) == 0:
            logger.warning("No work to do!")
            return {'completed': 0, 'failed': 0, 'skipped': skipped, 'total_cost': 0}

        # Execute transformations with concurrency control
        stats = {
            'completed': 0,
            'failed': 0,
            'skipped': skipped,
            'total_cost': 0.0
        }

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _transform_with_limit(func, model, task, strategy):
            async with semaphore:
                # Check budget before each transformation
                if stats['total_cost'] >= self.budget_limit:
                    logger.error(f"⚠️  BUDGET LIMIT REACHED: ${stats['total_cost']:.2f} / ${self.budget_limit:.2f}")
                    logger.error("Stopping transformations to prevent overspending!")
                    return None

                result = await self.transform_code(func, model, task, strategy)

                if result:
                    # Save to database immediately (checkpoint)
                    try:
                        with get_db_session() as session:
                            transformation = Transformation(
                                function_id=result['function_id'],
                                model=result['model'],
                                task=result['task'],
                                strategy=result['strategy'],
                                transformed_code=result['transformed_code'],
                                cyclomatic_complexity=result['cyclomatic_complexity'],
                                halstead_volume=result['halstead_volume'],
                                cost_usd=result['cost_usd'],
                                latency_ms=result['latency_ms'],
                                tokens_input=result['tokens_input'],
                                tokens_output=result['tokens_output'],
                                # SE metrics calculated later by SE calculator
                                preservation=None,
                                diversity=None,
                                effort=None,
                                se_score=None
                            )
                            session.add(transformation)
                            session.commit()

                        stats['completed'] += 1
                        stats['total_cost'] += result['cost_usd']

                    except Exception as e:
                        logger.error(f"Failed to save transformation to database: {e}")
                        stats['failed'] += 1
                else:
                    stats['failed'] += 1

                # Progress logging every 10 transformations
                total_done = stats['completed'] + stats['failed']
                if total_done % 10 == 0:
                    progress_pct = (total_done / len(work_queue)) * 100
                    logger.info(f"Progress: {total_done}/{len(work_queue)} ({progress_pct:.1f}%) | Cost: ${stats['total_cost']:.2f}")

                return result

        # Execute all transformations
        tasks_list = [
            _transform_with_limit(func, model, task, strategy)
            for func, model, task, strategy in work_queue
        ]

        await asyncio.gather(*tasks_list, return_exceptions=True)

        logger.success(f"\n✓ Transformation pipeline complete!")
        logger.info(f"Completed: {stats['completed']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Skipped (existing): {stats['skipped']}")
        logger.info(f"Total cost: ${stats['total_cost']:.2f}")

        return stats


async def main():
    """Main entry point for running transformations."""
    pipeline = TransformationPipeline(budget_limit_usd=2000.0)

    # Run transformations
    stats = await pipeline.run_transformations(
        models=None,  # Use all configured models
        tasks=['obfuscate'],  # Start with obfuscate only
        strategies=['zero_shot'],
        max_concurrent=5,
        resume=True  # Skip already-completed transformations
    )

    logger.info("=" * 60)
    logger.info("Transformation Statistics")
    logger.info("=" * 60)
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
