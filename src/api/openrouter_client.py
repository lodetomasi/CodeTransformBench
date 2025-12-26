"""
OpenRouter API client for CodeTransformBench.
Handles async requests with rate limiting, retries, and cost tracking.
"""

import asyncio
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
import aiohttp
from asyncio_throttle import Throttler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import config
from src.utils.logger import logger


class OpenRouterClient:
    """
    Async client for OpenRouter API with rate limiting and error handling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_requests_per_minute: int = None,
        max_retries: int = 3,
        timeout: int = 120
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (defaults to config)
            max_requests_per_minute: Rate limit (defaults to config)
            max_retries: Maximum retry attempts on failure
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or config.OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")

        self.base_url = config.OPENROUTER_BASE_URL
        self.max_requests_per_minute = max_requests_per_minute or config.MAX_REQUESTS_PER_MINUTE
        self.max_retries = max_retries
        self.timeout = timeout

        # Rate limiter: max_requests_per_minute requests per 60 seconds
        self.throttler = Throttler(rate_limit=self.max_requests_per_minute, period=60)

        # Track statistics
        self.total_requests = 0
        self.total_cost = 0.0
        self.total_tokens_input = 0
        self.total_tokens_output = 0

        logger.info(f"Initialized OpenRouter client (rate limit: {self.max_requests_per_minute} req/min)")

    async def _make_request(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        system: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make a single API request with rate limiting.

        Args:
            model: Model ID (e.g., 'anthropic/claude-3.5-sonnet')
            prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system: Optional system message

        Returns:
            API response dict
        """
        # Wait for rate limiter
        async with self.throttler:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://github.com/codetransformbench',  # Required by OpenRouter
                'X-Title': 'CodeTransformBench'  # Optional, helps OpenRouter
            }

            messages = []
            if system:
                messages.append({'role': 'system', 'content': system})
            messages.append({'role': 'user', 'content': prompt})

            data = {
                'model': model,
                'messages': messages,
                'temperature': temperature,
                'max_tokens': max_tokens
            }

            start_time = time.time()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{self.base_url}/chat/completions',
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    latency_ms = int((time.time() - start_time) * 1000)

                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")

                    result = await response.json()

                    # Extract usage and cost information
                    usage = result.get('usage', {})
                    tokens_input = usage.get('prompt_tokens', 0)
                    tokens_output = usage.get('completion_tokens', 0)

                    # Estimate cost (OpenRouter may not always return it)
                    # This is an approximation, actual cost may vary
                    cost_usd = self._estimate_cost(model, tokens_input, tokens_output)

                    # Update statistics
                    self.total_requests += 1
                    self.total_cost += cost_usd
                    self.total_tokens_input += tokens_input
                    self.total_tokens_output += tokens_output

                    return {
                        'content': result['choices'][0]['message']['content'],
                        'model': model,
                        'tokens_input': tokens_input,
                        'tokens_output': tokens_output,
                        'cost_usd': cost_usd,
                        'latency_ms': latency_ms,
                        'finish_reason': result['choices'][0].get('finish_reason')
                    }

    def _estimate_cost(self, model: str, tokens_input: int, tokens_output: int) -> float:
        """
        Estimate API call cost based on model and token usage.

        Args:
            model: Model ID
            tokens_input: Input tokens
            tokens_output: Output tokens

        Returns:
            Estimated cost in USD
        """
        # Get model cost from config
        from src.config import get_model_by_id

        try:
            model_info = get_model_by_id(model)
            cost_per_1k = model_info.get('cost_per_1k_tokens', 0.001)

            # Simple estimation: average of input/output cost
            # (Some models charge differently for input/output, but we simplify here)
            total_tokens = tokens_input + tokens_output
            cost = (total_tokens / 1000.0) * cost_per_1k

            return cost

        except Exception as e:
            logger.warning(f"Could not estimate cost for {model}: {e}")
            # Fallback: use average cost
            return ((tokens_input + tokens_output) / 1000.0) * 0.002

    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        system: Optional[str] = None,
        retry_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Generate completion with automatic retries.

        Args:
            model: Model ID
            prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system: Optional system message
            retry_on_error: Whether to retry on errors

        Returns:
            API response dict with content, cost, etc.
        """
        last_error = None

        for attempt in range(self.max_retries if retry_on_error else 1):
            try:
                result = await self._make_request(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system=system
                )

                if attempt > 0:
                    logger.info(f"Request succeeded on attempt {attempt + 1}")

                return result

            except asyncio.TimeoutError:
                last_error = "Request timeout"
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

            except aiohttp.ClientResponseError as e:
                # Fatal errors that should not be retried
                if e.status in [400, 401, 402, 403]:
                    error_msg = f"FATAL API error {e.status}: {e.message}"
                    if e.status == 402:
                        error_msg = "⚠️  INSUFFICIENT FUNDS on OpenRouter! Please add credits."
                    elif e.status == 401:
                        error_msg = "⚠️  INVALID API KEY! Check OPENROUTER_API_KEY."
                    logger.error(error_msg)
                    raise Exception(error_msg)  # Stop immediately, don't retry

                # Retryable errors
                elif e.status == 429:  # Rate limit
                    last_error = "Rate limit exceeded"
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(wait_time)
                else:  # 500, 503, etc. - server errors
                    last_error = f"HTTP {e.status}"
                    logger.error(f"API error: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)

            except Exception as e:
                last_error = str(e)
                logger.error(f"Request error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        # All retries failed
        raise Exception(f"All {self.max_retries} attempts failed. Last error: {last_error}")

    async def generate_batch(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = None
    ) -> List[Dict[str, Any]]:
        """
        Generate completions for multiple requests concurrently.

        Args:
            requests: List of request dicts with keys: model, prompt, temperature, max_tokens
            max_concurrent: Maximum concurrent requests (defaults to config)

        Returns:
            List of response dicts
        """
        max_concurrent = max_concurrent or config.MAX_CONCURRENT_REQUESTS

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _generate_with_semaphore(req: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await self.generate(**req)

        tasks = [_generate_with_semaphore(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error dicts
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {i} failed: {result}")
                processed_results.append({
                    'content': None,
                    'error': str(result),
                    'model': requests[i].get('model', 'unknown')
                })
            else:
                processed_results.append(result)

        return processed_results

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'total_requests': self.total_requests,
            'total_cost_usd': self.total_cost,
            'total_tokens_input': self.total_tokens_input,
            'total_tokens_output': self.total_tokens_output,
            'total_tokens': self.total_tokens_input + self.total_tokens_output,
            'avg_cost_per_request': self.total_cost / max(1, self.total_requests)
        }


async def test_client():
    """Test the OpenRouter client."""
    logger.info("Testing OpenRouter client...")

    client = OpenRouterClient()

    # Test single request
    try:
        result = await client.generate(
            model='anthropic/claude-3.5-sonnet',
            prompt='Write a Python function that calculates factorial. Respond with ONLY the code, no explanation.',
            temperature=0.2,
            max_tokens=500
        )

        logger.success(f"Test request successful!")
        logger.info(f"Generated {len(result['content'])} characters")
        logger.info(f"Cost: ${result['cost_usd']:.4f}")
        logger.info(f"Latency: {result['latency_ms']}ms")
        logger.info(f"Tokens: {result['tokens_input']} in + {result['tokens_output']} out")

        logger.info(f"\nGenerated code:\n{result['content'][:200]}...")

        # Print statistics
        stats = client.get_statistics()
        logger.info(f"\nClient statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Test failed: {e}")


if __name__ == '__main__':
    asyncio.run(test_client())
