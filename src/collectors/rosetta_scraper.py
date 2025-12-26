#!/usr/bin/env python3
"""
Rosetta Code scraper for CodeTransformBench.
Collects code samples across multiple languages from rosettacode.org.

Target: 260 functions stratified by language and complexity.
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import random

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import config
from src.collectors.metrics_calculator import calculate_metrics
from src.utils.logger import logger
from src.utils.progress import DataCollectionProgress


class RosettaCodeScraper:
    """Scraper for rosettacode.org."""

    BASE_URL = "https://rosettacode.org"
    TASKS_URL = f"{BASE_URL}/wiki/Category:Programming_Tasks"

    # Language mappings (Rosetta Code name → our name)
    LANGUAGE_MAP = {
        'Python': 'python',
        'Java': 'java',
        'JavaScript': 'javascript',
        'C++': 'cpp',
        'C%2B%2B': 'cpp'  # URL encoded version
    }

    # Target distribution
    TARGET = {
        'python': 65,   # 260 * 150/500 * 0.5 ≈ 65
        'java': 65,
        'javascript': 65,
        'cpp': 65
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize scraper.

        Args:
            cache_dir: Directory to cache HTML pages (default: data/raw/rosetta_html)
        """
        self.cache_dir = cache_dir or config.ROSETTA_HTML_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CodeTransformBench Research (lorenzo.detomasi@phd.research)'
        })

        self.collected_functions = []
        self.code_hashes = set()  # For deduplication

        logger.info(f"Initialized Rosetta Code scraper, cache: {self.cache_dir}")

    def _rate_limit(self):
        """Respect rate limit: 1 request per second."""
        time.sleep(1)

    def _fetch_url(self, url: str, use_cache: bool = True) -> Optional[str]:
        """
        Fetch URL with caching.

        Args:
            url: URL to fetch
            use_cache: Whether to use cached version if available

        Returns:
            HTML content or None if failed
        """
        # Generate cache filename
        cache_file = self.cache_dir / f"{hash(url) & 0xFFFFFFFF}.html"

        # Check cache
        if use_cache and cache_file.exists():
            logger.debug(f"Using cached: {url}")
            return cache_file.read_text(encoding='utf-8')

        # Fetch from web
        try:
            self._rate_limit()
            logger.debug(f"Fetching: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Cache response
            cache_file.write_text(response.text, encoding='utf-8')

            return response.text

        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def get_task_list(self) -> List[str]:
        """
        Get list of programming tasks from Rosetta Code.

        Returns:
            List of task URLs
        """
        logger.info("Fetching task list...")

        html = self._fetch_url(self.TASKS_URL)
        if not html:
            return []

        soup = BeautifulSoup(html, 'lxml')

        # Find all task links in the category page
        tasks = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Task links start with /wiki/ and are not categories or special pages
            if href.startswith('/wiki/') and ':' not in href and href != '/wiki/':
                task_url = self.BASE_URL + href
                if task_url not in tasks:
                    tasks.append(task_url)

        logger.info(f"Found {len(tasks)} tasks")
        return tasks

    def extract_code_from_task(self, task_url: str, task_name: str) -> List[Dict[str, Any]]:
        """
        Extract code samples for all languages from a task page.

        Args:
            task_url: URL of the task page
            task_name: Name of the task

        Returns:
            List of code samples with metadata
        """
        html = self._fetch_url(task_url)
        if not html:
            return []

        soup = BeautifulSoup(html, 'lxml')
        samples = []

        # Find code blocks for each language
        for lang_name, lang_code in self.LANGUAGE_MAP.items():
            # Look for language headers
            lang_header = soup.find('span', {'class': 'mw-headline', 'id': lang_name})
            if not lang_header:
                continue

            # Find code block after header
            current = lang_header.parent
            while current:
                current = current.find_next_sibling()
                if not current:
                    break

                # Stop at next language header
                if current.name in ['h2', 'h3'] and current.find('span', {'class': 'mw-headline'}):
                    break

                # Find <pre> or <code> tags
                code_blocks = current.find_all(['pre', 'code'])
                for block in code_blocks:
                    code = block.get_text()

                    # Clean up code
                    code = code.strip()
                    if len(code) < 50:  # Too short
                        continue
                    if len(code) > 5000:  # Too long
                        continue

                    samples.append({
                        'code': code,
                        'language': lang_code,
                        'task_name': task_name,
                        'source': 'rosetta_code',
                        'url': task_url
                    })

                    # Only take first code block per language
                    break

                if code_blocks:
                    break

        return samples

    def validate_code(self, code: str, language: str) -> bool:
        """
        Basic validation that code is syntactically reasonable.

        Args:
            code: Code to validate
            language: Programming language

        Returns:
            True if code seems valid
        """
        # Python: try to compile
        if language == 'python':
            try:
                compile(code, '<string>', 'exec')
                return True
            except:
                return False

        # Java: check for class definition
        if language == 'java':
            return 'class ' in code or 'interface ' in code

        # JavaScript: check for function definition
        if language == 'javascript':
            return 'function ' in code or '=>' in code or 'const ' in code or 'var ' in code

        # C++: check for basic syntax
        if language == 'cpp':
            return '#include' in code or 'int ' in code or 'void ' in code

        return True

    def collect_functions(self, target_per_language: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
        """
        Collect functions from Rosetta Code.

        Args:
            target_per_language: Target number of functions per language

        Returns:
            List of collected functions with metadata
        """
        if target_per_language is None:
            target_per_language = self.TARGET

        total_target = sum(target_per_language.values())
        logger.info(f"Target: {total_target} functions across {len(target_per_language)} languages")

        # Get task list
        tasks = self.get_task_list()
        random.shuffle(tasks)  # Randomize to get diverse samples

        # Track progress
        progress = DataCollectionProgress(total_target)
        collected_by_lang = defaultdict(list)

        # Collect from tasks
        for task_url in tasks:
            # Check if we've collected enough
            if all(len(collected_by_lang[lang]) >= target for lang, target in target_per_language.items()):
                break

            task_name = task_url.split('/')[-1].replace('_', ' ')
            logger.info(f"Processing task: {task_name}")

            # Extract code samples
            samples = self.extract_code_from_task(task_url, task_name)

            for sample in samples:
                lang = sample['language']

                # Skip if we have enough for this language
                if len(collected_by_lang[lang]) >= target_per_language.get(lang, 0):
                    continue

                code = sample['code']

                # Validate code
                if not self.validate_code(code, lang):
                    progress.add_failure()
                    logger.debug(f"Validation failed for {lang} in {task_name}")
                    continue

                # Calculate metrics
                try:
                    metrics = calculate_metrics(code, lang)
                except Exception as e:
                    logger.error(f"Metrics calculation failed: {e}")
                    progress.add_failure()
                    continue

                # Check for duplicates
                if metrics['code_hash'] in self.code_hashes:
                    progress.add_duplicate()
                    logger.debug(f"Duplicate skipped: {task_name}")
                    continue

                # Generate unique ID
                func_id = f"{lang}_rc_{len(collected_by_lang[lang]):03d}"

                # Create function record
                function = {
                    'id': func_id,
                    'language': lang,
                    'code': code,
                    'code_hash': metrics['code_hash'],
                    'cyclomatic_complexity': metrics['cyclomatic_complexity'],
                    'halstead_volume': metrics['halstead_volume'],
                    'lines_of_code': metrics['lines_of_code'],
                    'domain': self._infer_domain(task_name),
                    'source': 'rosetta_code',
                    'task_name': task_name,
                    'url': task_url
                }

                # Add to collection
                collected_by_lang[lang].append(function)
                self.code_hashes.add(metrics['code_hash'])
                progress.add_function(lang, metrics['complexity_tier'])

                logger.success(f"Collected {func_id}: {task_name} (CC={metrics['cyclomatic_complexity']})")

            # Print progress every 10 tasks
            if len(tasks) % 10 == 0:
                progress.print()

        # Flatten collected functions
        all_functions = []
        for lang_functions in collected_by_lang.values():
            all_functions.extend(lang_functions)

        logger.info(f"Collected {len(all_functions)} functions from Rosetta Code")
        progress.print()

        return all_functions

    def _infer_domain(self, task_name: str) -> str:
        """
        Infer domain from task name.

        Args:
            task_name: Name of the task

        Returns:
            Domain classification
        """
        task_lower = task_name.lower()

        if any(kw in task_lower for kw in ['sort', 'search', 'tree', 'graph', 'dynamic']):
            return 'algorithms'
        if any(kw in task_lower for kw in ['list', 'array', 'stack', 'queue', 'hash']):
            return 'data_structures'
        if any(kw in task_lower for kw in ['string', 'text', 'parse', 'regex']):
            return 'strings'
        if any(kw in task_lower for kw in ['math', 'number', 'prime', 'factorial', 'fibonacci']):
            return 'math'
        if any(kw in task_lower for kw in ['file', 'read', 'write', 'io']):
            return 'io'

        return 'other'

    def save_to_jsonl(self, functions: List[Dict[str, Any]], output_path: Path):
        """Save functions to JSONL file."""
        with open(output_path, 'w') as f:
            for func in functions:
                f.write(json.dumps(func) + '\n')

        logger.success(f"Saved {len(functions)} functions to {output_path}")


def main():
    """Main entry point for Rosetta Code scraper."""
    logger.info("Starting Rosetta Code scraper...")

    scraper = RosettaCodeScraper()
    functions = scraper.collect_functions()

    # Save to file
    output_path = config.DATA_PROCESSED_DIR / 'rosetta_code_functions.jsonl'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scraper.save_to_jsonl(functions, output_path)

    logger.success(f"✓ Rosetta Code collection complete: {len(functions)} functions")

    # Print summary
    by_lang = defaultdict(int)
    by_tier = defaultdict(int)
    for func in functions:
        by_lang[func['language']] += 1
        cc = func['cyclomatic_complexity']
        if cc <= 10:
            by_tier['simple'] += 1
        elif cc <= 30:
            by_tier['medium'] += 1
        else:
            by_tier['complex'] += 1

    logger.info("Summary by language:")
    for lang, count in sorted(by_lang.items()):
        logger.info(f"  {lang}: {count}")

    logger.info("Summary by complexity:")
    for tier, count in sorted(by_tier.items()):
        logger.info(f"  {tier}: {count}")


if __name__ == '__main__':
    main()
