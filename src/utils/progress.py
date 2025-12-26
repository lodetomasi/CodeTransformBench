"""
Progress monitoring for CodeTransformBench.
Beautiful terminal dashboards using Rich library.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.live import Live
from rich.layout import Layout

console = Console()


class ExperimentProgress:
    """
    Progress tracker for transformation experiments.

    Displays live dashboard with:
    - Current tier and model
    - Progress (completed/total)
    - Success rate
    - Average SE score
    - Cost tracking
    - Rate limiting info
    - ETA
    """

    def __init__(self, tier: str, model_name: str, total_experiments: int):
        self.tier = tier
        self.model_name = model_name
        self.total_experiments = total_experiments
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.total_cost = 0.0
        self.se_scores = []
        self.start_time = datetime.now()
        self.requests_per_minute = 0

    def update(
        self,
        success: bool,
        se_score: Optional[float] = None,
        cost: float = 0.0
    ):
        """Update progress with experiment result."""
        self.completed += 1

        if success:
            self.successful += 1
            if se_score is not None:
                self.se_scores.append(se_score)
        else:
            self.failed += 1

        self.total_cost += cost

        # Calculate requests per minute
        elapsed = (datetime.now() - self.start_time).total_seconds() / 60
        if elapsed > 0:
            self.requests_per_minute = self.completed / elapsed

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.completed == 0:
            return 0.0
        return (self.successful / self.completed) * 100

    @property
    def avg_se(self) -> float:
        """Get average SE score."""
        if not self.se_scores:
            return 0.0
        return sum(self.se_scores) / len(self.se_scores)

    @property
    def std_se(self) -> float:
        """Get standard deviation of SE scores."""
        if len(self.se_scores) < 2:
            return 0.0
        mean = self.avg_se
        variance = sum((x - mean) ** 2 for x in self.se_scores) / len(self.se_scores)
        return variance ** 0.5

    @property
    def eta(self) -> str:
        """Get estimated time to completion."""
        if self.completed == 0:
            return "calculating..."

        elapsed = datetime.now() - self.start_time
        avg_time_per_experiment = elapsed / self.completed
        remaining_experiments = self.total_experiments - self.completed
        eta_delta = avg_time_per_experiment * remaining_experiments

        # Format as hours:minutes
        hours, remainder = divmod(int(eta_delta.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{hours}h {minutes}m"

    def render(self) -> Panel:
        """Render progress as Rich panel."""
        progress_pct = (self.completed / self.total_experiments * 100) if self.total_experiments > 0 else 0

        content = f"""[bold]CodeTransformBench - {self.tier.upper()}[/bold]

[cyan]Model:[/cyan] {self.model_name}
[cyan]Progress:[/cyan] {self.completed}/{self.total_experiments} ({progress_pct:.1f}%)
[cyan]Success:[/cyan] {self.success_rate:.1f}%
[cyan]Avg SE:[/cyan] {self.avg_se:.2f} (σ={self.std_se:.2f})
[cyan]Cost:[/cyan] ${self.total_cost:.2f}
[cyan]Rate:[/cyan] {self.requests_per_minute:.0f} req/min
[cyan]ETA:[/cyan] {self.eta}
"""

        return Panel(
            content,
            border_style="green" if self.success_rate > 70 else "yellow" if self.success_rate > 50 else "red",
            padding=(1, 2)
        )

    def print(self):
        """Print current progress to console."""
        console.print(self.render())


class DataCollectionProgress:
    """
    Progress tracker for data collection phase.

    Tracks scraping from Rosetta Code and TheAlgorithms.
    """

    def __init__(self, target_functions: int):
        self.target = target_functions
        self.collected = 0
        self.by_language = {'python': 0, 'java': 0, 'javascript': 0, 'cpp': 0}
        self.by_complexity = {'simple': 0, 'medium': 0, 'complex': 0}
        self.duplicates_skipped = 0
        self.validation_failures = 0

    def add_function(self, language: str, complexity_tier: str):
        """Record a successfully collected function."""
        self.collected += 1
        self.by_language[language] = self.by_language.get(language, 0) + 1
        self.by_complexity[complexity_tier] = self.by_complexity.get(complexity_tier, 0) + 1

    def add_duplicate(self):
        """Record a duplicate that was skipped."""
        self.duplicates_skipped += 1

    def add_failure(self):
        """Record a validation failure."""
        self.validation_failures += 1

    def render(self) -> Table:
        """Render progress as Rich table."""
        table = Table(title="Data Collection Progress", show_header=True, header_style="bold cyan")

        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Total Collected", f"{self.collected}/{self.target}")
        table.add_row("Progress", f"{self.collected/self.target*100:.1f}%")
        table.add_row("", "")

        table.add_row("Python", str(self.by_language.get('python', 0)))
        table.add_row("Java", str(self.by_language.get('java', 0)))
        table.add_row("JavaScript", str(self.by_language.get('javascript', 0)))
        table.add_row("C++", str(self.by_language.get('cpp', 0)))
        table.add_row("", "")

        table.add_row("Simple (CC≤10)", str(self.by_complexity.get('simple', 0)))
        table.add_row("Medium (CC 11-30)", str(self.by_complexity.get('medium', 0)))
        table.add_row("Complex (CC≥31)", str(self.by_complexity.get('complex', 0)))
        table.add_row("", "")

        table.add_row("Duplicates Skipped", str(self.duplicates_skipped))
        table.add_row("Validation Failures", str(self.validation_failures))

        return table

    def print(self):
        """Print current progress to console."""
        console.print(self.render())


def create_progress_bar(description: str, total: int) -> Progress:
    """
    Create a simple progress bar.

    Usage:
        with create_progress_bar("Processing functions", 500) as progress:
            task = progress.add_task("Processing", total=500)
            for i in range(500):
                # Do work
                progress.update(task, advance=1)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    )


if __name__ == '__main__':
    # Test progress displays
    import time

    # Test experiment progress
    print("\n=== Experiment Progress Test ===\n")
    exp_progress = ExperimentProgress("tier1", "Llama 3.1 8B", 100)

    for i in range(20):
        success = i % 3 != 0  # 67% success rate
        se = 6.5 + (i % 5) if success else None
        exp_progress.update(success, se, 0.001)

        if i % 5 == 0:
            exp_progress.print()
            time.sleep(0.1)

    # Test data collection progress
    print("\n=== Data Collection Progress Test ===\n")
    data_progress = DataCollectionProgress(500)

    languages = ['python', 'java', 'javascript', 'cpp']
    tiers = ['simple', 'medium', 'complex']

    for i in range(50):
        data_progress.add_function(
            languages[i % len(languages)],
            tiers[i % len(tiers)]
        )

        if i % 10 == 0:
            data_progress.add_duplicate()

        if i % 15 == 0:
            data_progress.add_failure()

    data_progress.print()

    # Test simple progress bar
    print("\n=== Simple Progress Bar Test ===\n")
    with create_progress_bar("Processing", 50) as progress:
        task = progress.add_task("Processing items", total=50)
        for i in range(50):
            time.sleep(0.02)
            progress.update(task, advance=1)

    console.print("\n[bold green]✓ All progress displays working![/bold green]")
