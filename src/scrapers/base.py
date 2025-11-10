"""Base scraper with rate limiting and retry logic."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from loguru import logger
from pydantic import BaseModel, Field, validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class PriceData(BaseModel):
    """Validated price data model."""

    product_id: str = Field(..., min_length=1)
    product_name: str = Field(..., min_length=1)
    price: float = Field(..., gt=0)
    original_price: float | None = Field(None, gt=0)
    currency: str = Field(default="USD")
    category: str = Field(..., min_length=1)
    subcategory: str | None = None
    retailer: str = Field(..., min_length=1)
    url: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    unit: str | None = None
    quantity: float | None = Field(None, gt=0)
    in_stock: bool = True
    rating: float | None = Field(None, ge=0, le=5)
    review_count: int | None = Field(None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @validator("price", "original_price", pre=True)
    def clean_price(cls, v):
        """Clean and validate price values."""
        if v is None:
            return None
        if isinstance(v, str):
            # Remove currency symbols and whitespace
            cleaned = v.replace("$", "").replace(",", "").strip()
            return float(cleaned)
        return float(v)

    @validator("url")
    def validate_url(cls, v):
        """Validate URL format."""
        parsed = urlparse(v)
        if not all([parsed.scheme, parsed.netloc]):
            raise ValueError("Invalid URL format")
        return v

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


@dataclass
class RateLimiter:
    """Simple rate limiter for request throttling."""

    requests_per_second: float = 1.0
    last_request_time: float = field(default=0.0)
    _lock: Any = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize the rate limiter."""
        import threading

        self._lock = threading.Lock()

    def wait(self):
        """Wait if necessary to respect rate limit."""
        with self._lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            min_interval = 1.0 / self.requests_per_second

            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)

            self.last_request_time = time.time()


class ScraperConfig(BaseModel):
    """Configuration for price scrapers."""

    rate_limit: float = Field(default=1.0, gt=0)
    max_retries: int = Field(default=3, ge=1)
    timeout: int = Field(default=30, gt=0)
    user_agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    )
    proxy: str | None = None
    verify_ssl: bool = True
    headers: dict[str, str] = Field(default_factory=dict)


class PriceScraper(ABC):
    """Abstract base class for all price scrapers."""

    def __init__(self, config=None):
        """Initialize the scraper."""
        self.config = config or ScraperConfig()
        self.rate_limiter = RateLimiter(requests_per_second=self.config.rate_limit)
        self.session = self._create_session()
        self._setup_logging()

    def _create_session(self):
        """Create and configure requests session."""
        session = requests.Session()

        # Set default headers
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
        headers.update(self.config.headers)
        session.headers.update(headers)

        # Set proxy if configured
        if self.config.proxy:
            session.proxies = {
                "http": self.config.proxy,
                "https": self.config.proxy,
            }

        return session

    def _setup_logging(self):
        """Configure logging for the scraper."""
        logger.info(
            f"Initialized {self.__class__.__name__} with rate limit "
            f"{self.config.rate_limit} req/s"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.RequestException, requests.Timeout)),
    )
    def _fetch_page(self, url):
        """Fetch and parse a web page with retry logic."""
        self.rate_limiter.wait()

        logger.debug(f"Fetching URL: {url}")

        response = self.session.get(
            url,
            timeout=self.config.timeout,
            verify=self.config.verify_ssl,
        )
        response.raise_for_status()

        return BeautifulSoup(response.content, "lxml")

    def validate_price_data(self, data):
        """Validate scraped data and return PriceData or None."""
        try:
            return PriceData(**data)
        except Exception as e:
            logger.warning(f"Data validation failed: {e}")
            return None

    @property
    @abstractmethod
    def retailer_name(self):
        """Return the retailer name."""
        pass

    @property
    @abstractmethod
    def base_url(self):
        """Return the base URL for the retailer."""
        pass

    @abstractmethod
    def scrape_product(self, url):
        """Scrape price data for a single product."""
        pass

    @abstractmethod
    def scrape_category(self, category, max_products=100):
        """Scrape prices for all products in a category."""
        pass

    @abstractmethod
    def get_categories(self):
        """Get available product categories."""
        pass

    def close(self):
        """Close the scraper session."""
        self.session.close()
        logger.info(f"Closed {self.__class__.__name__} session")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
