#!/usr/bin/env python3
"""
TheAlgorithms repository cloner for CodeTransformBench.
Clones repositories and extracts function-level code samples.

Target: 240 functions stratified by language and complexity.
"""

import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
import random
import re
import ast

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import config
from src.collectors.metrics_calculator import calculate_metrics
from src.utils.logger import logger
from src.utils.progress import DataCollectionProgress


class TheAlgorithmsCloner:
    """Cloner for TheAlgorithms repositories."""

    REPOS = {
        'python': 'https://github.com/TheAlgorithms/Python.git',
        'java': 'https://github.com/TheAlgorithms/Java.git',
        'javascript': 'https://github.com/TheAlgorithms/JavaScript.git',
        'cpp': 'https://github.com/TheAlgorithms/C-Plus-Plus.git'
    }

    # Target distribution
    TARGET = {
        'python': 60,   # 240 * 150/500 * 0.5 ≈ 60
        'java': 60,
        'javascript': 60,
        'cpp': 60
    }

    FILE_EXTENSIONS = {
        'python': '.py',
        'java': '.java',
        'javascript': '.js',
        'cpp': '.cpp'
    }

    def __init__(self, clone_dir: Optional[Path] = None, existing_hashes: Optional[Set[str]] = None):
        """
        Initialize cloner.

        Args:
            clone_dir: Directory to clone repos (default: data/raw/algorithms_repos)
            existing_hashes: Set of existing code hashes to avoid duplicates
        """
        self.clone_dir = clone_dir or config.ALGORITHMS_REPOS_DIR
        self.clone_dir.mkdir(parents=True, exist_ok=True)

        self.existing_hashes = existing_hashes or set()
        self.collected_functions = []

        logger.info(f"Initialized TheAlgorithms cloner, clone dir: {self.clone_dir}")

    def clone_repo(self, language: str, url: str) -> Optional[Path]:
        """
        Clone a repository (shallow clone for speed).

        Args:
            language: Language name
            url: Repository URL

        Returns:
            Path to cloned repository or None if failed
        """
        repo_path = self.clone_dir / language

        # Skip if already cloned
        if repo_path.exists() and (repo_path / '.git').exists():
            logger.info(f"Repository already cloned: {language}")
            return repo_path

        logger.info(f"Cloning {language} repository...")

        try:
            subprocess.run(
                ['git', 'clone', '--depth', '1', url, str(repo_path)],
                check=True,
                capture_output=True,
                timeout=300
            )
            logger.success(f"Cloned {language} repository")
            return repo_path

        except subprocess.TimeoutExpired:
            logger.error(f"Clone timeout for {language}")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"Clone failed for {language}: {e}")
            return None

    def find_source_files(self, repo_path: Path, language: str) -> List[Path]:
        """
        Find all source files in repository.

        Args:
            repo_path: Path to repository
            language: Language name

        Returns:
            List of source file paths
        """
        extension = self.FILE_EXTENSIONS[language]
        files = list(repo_path.rglob(f'*{extension}'))

        # Filter out test files, __init__.py, etc.
        filtered = []
        for f in files:
            name = f.name.lower()

            # Skip test files
            if 'test' in name or 'spec' in name:
                continue

            # Skip __init__.py
            if name == '__init__.py':
                continue

            # Skip files in test directories
            if any(part.lower() in ['test', 'tests', '__pycache__'] for part in f.parts):
                continue

            # Check file size (100 bytes to 5KB)
            try:
                size = f.stat().st_size
                if 100 <= size <= 5000:
                    filtered.append(f)
            except:
                continue

        logger.info(f"Found {len(filtered)} source files for {language}")
        return filtered

    def extract_python_functions(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract function definitions from Python file."""
        try:
            code = file_path.read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(code)

            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Get function source code
                    func_lines = code.split('\n')[node.lineno - 1:node.end_lineno]
                    func_code = '\n'.join(func_lines)

                    # Skip very short or very long functions
                    if len(func_code) < 50 or len(func_code) > 3000:
                        continue

                    functions.append({
                        'code': func_code,
                        'name': node.name,
                        'file': str(file_path)
                    })

            return functions

        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return []

    def extract_java_functions(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract method definitions from Java file (simple heuristic)."""
        try:
            code = file_path.read_text(encoding='utf-8', errors='ignore')

            # Simple regex to find methods
            # Matches: [modifiers] returnType methodName(params) {
            pattern = r'(public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*\{'

            functions = []
            for match in re.finditer(pattern, code):
                # Extract method body (simplified - just take next ~1000 chars)
                start = match.start()
                end = min(start + 2000, len(code))
                func_code = code[start:end]

                # Try to find matching closing brace
                brace_count = 0
                for i, char in enumerate(func_code):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            func_code = func_code[:i + 1]
                            break

                if 50 <= len(func_code) <= 3000:
                    functions.append({
                        'code': func_code,
                        'name': match.group(2),
                        'file': str(file_path)
                    })

            return functions

        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return []

    def extract_javascript_functions(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract function definitions from JavaScript file (simple heuristic)."""
        try:
            code = file_path.read_text(encoding='utf-8', errors='ignore')

            functions = []

            # Pattern 1: function name() {}
            pattern1 = r'function\s+(\w+)\s*\([^)]*\)\s*\{'

            # Pattern 2: const name = () => {}
            pattern2 = r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*\{'

            for pattern in [pattern1, pattern2]:
                for match in re.finditer(pattern, code):
                    start = match.start()
                    end = min(start + 2000, len(code))
                    func_code = code[start:end]

                    # Find closing brace
                    brace_count = 0
                    for i, char in enumerate(func_code):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                func_code = func_code[:i + 1]
                                break

                    if 50 <= len(func_code) <= 3000:
                        functions.append({
                            'code': func_code,
                            'name': match.group(1),
                            'file': str(file_path)
                        })

            return functions

        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return []

    def extract_cpp_functions(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract function definitions from C++ file (simple heuristic)."""
        try:
            code = file_path.read_text(encoding='utf-8', errors='ignore')

            # Simple pattern: returnType functionName(params) {
            pattern = r'[\w:<>*&]+\s+(\w+)\s*\([^)]*\)\s*\{'

            functions = []
            for match in re.finditer(pattern, code):
                start = match.start()
                end = min(start + 2000, len(code))
                func_code = code[start:end]

                # Find closing brace
                brace_count = 0
                for i, char in enumerate(func_code):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            func_code = func_code[:i + 1]
                            break

                if 50 <= len(func_code) <= 3000:
                    functions.append({
                        'code': func_code,
                        'name': match.group(1),
                        'file': str(file_path)
                    })

            return functions

        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return []

    def extract_functions_from_file(self, file_path: Path, language: str) -> List[Dict[str, Any]]:
        """Extract functions from a source file."""
        if language == 'python':
            return self.extract_python_functions(file_path)
        elif language == 'java':
            return self.extract_java_functions(file_path)
        elif language == 'javascript':
            return self.extract_javascript_functions(file_path)
        elif language == 'cpp':
            return self.extract_cpp_functions(file_path)
        else:
            return []

    def collect_functions(self, target_per_language: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
        """
        Collect functions from TheAlgorithms repositories.

        Args:
            target_per_language: Target number of functions per language

        Returns:
            List of collected functions with metadata
        """
        if target_per_language is None:
            target_per_language = self.TARGET

        total_target = sum(target_per_language.values())
        logger.info(f"Target: {total_target} functions across {len(target_per_language)} languages")

        progress = DataCollectionProgress(total_target)
        collected_by_lang = defaultdict(list)

        # Process each language
        for language, repo_url in self.REPOS.items():
            target = target_per_language.get(language, 0)
            if target == 0:
                continue

            logger.info(f"\n=== Processing {language} ===")

            # Clone repository
            repo_path = self.clone_repo(language, repo_url)
            if not repo_path:
                continue

            # Find source files
            source_files = self.find_source_files(repo_path, language)
            random.shuffle(source_files)

            # Extract functions from files
            for file_path in source_files:
                if len(collected_by_lang[language]) >= target:
                    break

                functions = self.extract_functions_from_file(file_path, language)

                for func_data in functions:
                    if len(collected_by_lang[language]) >= target:
                        break

                    code = func_data['code']

                    # Calculate metrics
                    try:
                        metrics = calculate_metrics(code, language)
                    except Exception as e:
                        logger.debug(f"Metrics calculation failed: {e}")
                        progress.add_failure()
                        continue

                    # Quality filters
                    cc = metrics['cyclomatic_complexity']
                    if cc < 3:  # Too simple
                        continue

                    # Check for duplicates
                    if metrics['code_hash'] in self.existing_hashes:
                        progress.add_duplicate()
                        continue

                    # Generate unique ID
                    func_id = f"{language}_ta_{len(collected_by_lang[language]):03d}"

                    # Infer domain from file path
                    domain = self._infer_domain_from_path(file_path)

                    # Create function record
                    function = {
                        'id': func_id,
                        'language': language,
                        'code': code,
                        'code_hash': metrics['code_hash'],
                        'cyclomatic_complexity': cc,
                        'halstead_volume': metrics['halstead_volume'],
                        'lines_of_code': metrics['lines_of_code'],
                        'domain': domain,
                        'source': 'the_algorithms',
                        'task_name': func_data['name'],
                        'file_path': func_data['file']
                    }

                    # Add to collection
                    collected_by_lang[language].append(function)
                    self.existing_hashes.add(metrics['code_hash'])
                    progress.add_function(language, metrics['complexity_tier'])

                    logger.success(f"Collected {func_id}: {func_data['name']} (CC={cc})")

            logger.info(f"Collected {len(collected_by_lang[language])} functions for {language}")

        # Flatten collected functions
        all_functions = []
        for lang_functions in collected_by_lang.values():
            all_functions.extend(lang_functions)

        logger.info(f"\nCollected {len(all_functions)} functions from TheAlgorithms")
        progress.print()

        return all_functions

    def _infer_domain_from_path(self, file_path: Path) -> str:
        """Infer domain from file path."""
        path_str = str(file_path).lower()

        if any(kw in path_str for kw in ['sort', 'search', 'dp', 'dynamic', 'graph', 'tree']):
            return 'algorithms'
        if any(kw in path_str for kw in ['data_structure', 'list', 'stack', 'queue', 'hash']):
            return 'data_structures'
        if any(kw in path_str for kw in ['string', 'text']):
            return 'strings'
        if any(kw in path_str for kw in ['math', 'number']):
            return 'math'
        if any(kw in path_str for kw in ['file', 'io']):
            return 'io'

        return 'other'

    def save_to_jsonl(self, functions: List[Dict[str, Any]], output_path: Path):
        """Save functions to JSONL file."""
        with open(output_path, 'w') as f:
            for func in functions:
                f.write(json.dumps(func) + '\n')

        logger.success(f"Saved {len(functions)} functions to {output_path}")


def main():
    """Main entry point for TheAlgorithms cloner."""
    logger.info("Starting TheAlgorithms cloner...")

    cloner = TheAlgorithmsCloner()
    functions = cloner.collect_functions()

    # Save to file
    output_path = config.DATA_PROCESSED_DIR / 'algorithms_functions.jsonl'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cloner.save_to_jsonl(functions, output_path)

    logger.success(f"✓ TheAlgorithms collection complete: {len(functions)} functions")

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
