"""Amazon and Walmart scrapers (mock implementations)."""

import random
import re
from datetime import datetime

from loguru import logger

from src.scrapers.base import PriceScraper

# CPI category mappings for normalization
CPI_CATEGORIES = {
    "grocery": {
        "weight": 0.143,
        "subcategories": [
            "cereals",
            "bakery",
            "meats",
            "poultry",
            "fish",
            "eggs",
            "dairy",
            "fruits",
            "vegetables",
            "beverages",
            "snacks",
        ],
    },
    "housing": {
        "weight": 0.424,
        "subcategories": [
            "rent",
            "utilities",
            "furniture",
            "appliances",
            "housekeeping",
        ],
    },
    "apparel": {
        "weight": 0.026,
        "subcategories": [
            "mens_clothing",
            "womens_clothing",
            "childrens_clothing",
            "footwear",
            "accessories",
        ],
    },
    "transportation": {
        "weight": 0.160,
        "subcategories": [
            "vehicles",
            "gasoline",
            "maintenance",
            "insurance",
            "public_transit",
        ],
    },
    "medical": {
        "weight": 0.085,
        "subcategories": [
            "medical_services",
            "drugs",
            "medical_equipment",
            "insurance",
        ],
    },
    "recreation": {
        "weight": 0.054,
        "subcategories": [
            "electronics",
            "sports",
            "toys",
            "pets",
            "entertainment",
        ],
    },
    "education": {
        "weight": 0.062,
        "subcategories": [
            "tuition",
            "books",
            "supplies",
            "childcare",
        ],
    },
    "other": {
        "weight": 0.046,
        "subcategories": [
            "personal_care",
            "tobacco",
            "miscellaneous",
        ],
    },
}


class AmazonScraper(PriceScraper):
    """Mock Amazon scraper. Note: Amazon's robots.txt restricts real scraping."""

    def __init__(self, config=None):
        """Initialize Amazon scraper."""
        super().__init__(config)
        self._mock_products = self._generate_mock_catalog()

    @property
    def retailer_name(self):
        return "Amazon"

    @property
    def base_url(self):
        return "https://www.amazon.com"

    def _generate_mock_catalog(self):
        """Generate mock product catalog for testing."""
        catalog = {}

        mock_products = {
            "grocery": [
                ("B001EO5Q64", "Organic Whole Milk 1 Gallon", 6.99, "dairy"),
                ("B07FWML5SZ", "Free Range Eggs 12ct", 5.49, "eggs"),
                ("B01N5KUWDI", "Whole Wheat Bread Loaf", 3.99, "bakery"),
                ("B00BPXEVG4", "Ground Beef 1lb 80/20", 6.99, "meats"),
                ("B074H6M7FL", "Organic Bananas 2lb", 2.49, "fruits"),
                ("B07PLGRTZ9", "Mixed Vegetables Frozen 16oz", 3.29, "vegetables"),
                ("B001EO5Q65", "Orange Juice 64oz", 4.99, "beverages"),
                ("B001EO5Q66", "Cheerios Cereal 18oz", 5.49, "cereals"),
            ],
            "apparel": [
                ("B07N1W2QXB", "Men's Cotton T-Shirt", 19.99, "mens_clothing"),
                ("B08L61W5KZ", "Women's Running Shoes", 79.99, "footwear"),
                ("B07KWL93MZ", "Kids' Winter Jacket", 49.99, "childrens_clothing"),
            ],
            "electronics": [
                ("B09B8V1LZ3", "Wireless Bluetooth Headphones", 79.99, "electronics"),
                ("B08N5WRWNW", "4K Smart TV 55 inch", 449.99, "electronics"),
                ("B09V3KXJPB", "Laptop Computer 15.6 inch", 699.99, "electronics"),
            ],
            "household": [
                ("B07GJBBGHG", "Laundry Detergent 150oz", 19.99, "housekeeping"),
                ("B0755SKTFQ", "Paper Towels 12 Rolls", 24.99, "housekeeping"),
                ("B01KHGDLSQ", "Air Fryer 5.8 Qt", 89.99, "appliances"),
            ],
        }

        for category, products in mock_products.items():
            catalog[category] = [
                {
                    "product_id": pid,
                    "product_name": name,
                    "price": price,
                    "subcategory": subcat,
                }
                for pid, name, price, subcat in products
            ]

        return catalog

    def _normalize_category(self, amazon_category):
        """Map Amazon category to CPI category."""
        category_map = {
            "grocery": "grocery",
            "food": "grocery",
            "apparel": "apparel",
            "clothing": "apparel",
            "electronics": "recreation",
            "household": "housing",
            "home": "housing",
            "health": "medical",
            "beauty": "other",
            "toys": "recreation",
            "books": "education",
            "automotive": "transportation",
        }
        return category_map.get(amazon_category.lower(), "other")

    def _normalize_price(self, price, unit=None, quantity=None):
        """Normalize price to standard unit for comparison."""
        if unit and quantity:
            # Convert to per-unit price
            unit_multipliers = {
                "oz": 16.0,  # Convert to per-pound
                "lb": 1.0,
                "gal": 1.0,
                "qt": 4.0,  # Convert to per-gallon
                "ct": 1.0,
            }
            multiplier = unit_multipliers.get(unit.lower(), 1.0)
            return price * multiplier / quantity
        return price

    def scrape_product(self, url):
        """Scrape a single product page (mock implementation)."""
        logger.info(f"Scraping Amazon product: {url}")

        # Mock implementation - extract ASIN from URL
        asin_match = re.search(r"/dp/([A-Z0-9]{10})", url)
        if not asin_match:
            logger.warning(f"Could not extract ASIN from URL: {url}")
            return None

        asin = asin_match.group(1)

        # Search for product in mock catalog
        for category, products in self._mock_products.items():
            for product in products:
                if product["product_id"] == asin:
                    # Add some random price variation to simulate real-time changes
                    price_variation = random.uniform(-0.05, 0.05)
                    current_price = product["price"] * (1 + price_variation)

                    data = {
                        "product_id": product["product_id"],
                        "product_name": product["product_name"],
                        "price": round(current_price, 2),
                        "category": self._normalize_category(category),
                        "subcategory": product["subcategory"],
                        "retailer": self.retailer_name,
                        "url": url,
                        "timestamp": datetime.utcnow(),
                        "in_stock": random.random() > 0.1,  # 90% in stock
                        "rating": round(random.uniform(3.5, 5.0), 1),
                        "review_count": random.randint(10, 10000),
                    }

                    return self.validate_price_data(data)

        logger.warning(f"Product not found in catalog: {asin}")
        return None

    def scrape_category(self, category, max_products=100):
        """Scrape all products in a category."""
        logger.info(f"Scraping Amazon category: {category}")

        products = []
        mock_products = self._mock_products.get(category.lower(), [])

        for product in mock_products[:max_products]:
            # Simulate price variation
            price_variation = random.uniform(-0.05, 0.05)
            current_price = product["price"] * (1 + price_variation)

            data = {
                "product_id": product["product_id"],
                "product_name": product["product_name"],
                "price": round(current_price, 2),
                "category": self._normalize_category(category),
                "subcategory": product["subcategory"],
                "retailer": self.retailer_name,
                "url": f"{self.base_url}/dp/{product['product_id']}",
                "timestamp": datetime.utcnow(),
                "in_stock": random.random() > 0.1,
            }

            validated = self.validate_price_data(data)
            if validated:
                products.append(validated)

        logger.info(f"Scraped {len(products)} products from Amazon/{category}")
        return products

    def get_categories(self):
        """Get available categories."""
        return list(self._mock_products.keys())


class WalmartScraper(PriceScraper):
    """Mock Walmart price scraper."""

    def __init__(self, config=None):
        """Initialize Walmart scraper."""
        super().__init__(config)
        self._mock_products = self._generate_mock_catalog()

    @property
    def retailer_name(self):
        return "Walmart"

    @property
    def base_url(self):
        return "https://www.walmart.com"

    def _generate_mock_catalog(self):
        """Generate mock product catalog for testing."""
        catalog = {}

        mock_products = {
            "grocery": [
                ("100012345", "Great Value Milk 1 Gallon", 3.78, "dairy"),
                ("100012346", "Great Value Large Eggs 12ct", 3.24, "eggs"),
                ("100012347", "Wonder Bread White 20oz", 2.48, "bakery"),
                ("100012348", "Ground Beef 1lb 73/27", 5.47, "meats"),
                ("100012349", "Fresh Bananas 1lb", 0.58, "fruits"),
                ("100012350", "Frozen Mixed Vegetables 12oz", 1.28, "vegetables"),
                ("100012351", "Tropicana Orange Juice 52oz", 3.98, "beverages"),
                ("100012352", "Frosted Flakes Cereal 13.5oz", 3.98, "cereals"),
                ("100012353", "Chicken Breast Boneless 1lb", 4.97, "poultry"),
            ],
            "apparel": [
                ("200012345", "Hanes Men's T-Shirt Pack", 14.97, "mens_clothing"),
                ("200012346", "Athletic Works Running Shoes", 24.97, "footwear"),
                ("200012347", "George Women's Cardigan", 19.97, "womens_clothing"),
            ],
            "electronics": [
                ("300012345", "onn. Wireless Earbuds", 19.88, "electronics"),
                ("300012346", "TCL 50 inch 4K Roku TV", 248.00, "electronics"),
                ("300012347", "HP Laptop 15.6 inch", 449.00, "electronics"),
            ],
            "household": [
                ("400012345", "Tide Laundry Detergent 92oz", 12.97, "housekeeping"),
                ("400012346", "Bounty Paper Towels 8 Rolls", 15.97, "housekeeping"),
                ("400012347", "Ninja Air Fryer 4 Qt", 69.00, "appliances"),
            ],
            "personal_care": [
                ("500012345", "Crest Toothpaste 4.6oz", 3.97, "personal_care"),
                ("500012346", "Dove Body Wash 22oz", 6.97, "personal_care"),
                ("500012347", "Advil Pain Reliever 100ct", 9.97, "medical_equipment"),
            ],
        }

        for category, products in mock_products.items():
            catalog[category] = [
                {
                    "product_id": pid,
                    "product_name": name,
                    "price": price,
                    "subcategory": subcat,
                }
                for pid, name, price, subcat in products
            ]

        return catalog

    def _normalize_category(self, walmart_category):
        """Map Walmart category to CPI category."""
        category_map = {
            "grocery": "grocery",
            "food": "grocery",
            "apparel": "apparel",
            "clothing": "apparel",
            "electronics": "recreation",
            "household": "housing",
            "home": "housing",
            "health": "medical",
            "personal_care": "other",
            "beauty": "other",
            "toys": "recreation",
            "books": "education",
            "automotive": "transportation",
            "pharmacy": "medical",
        }
        return category_map.get(walmart_category.lower(), "other")

    def scrape_product(self, url):
        """Scrape a single product page (mock implementation)."""
        logger.info(f"Scraping Walmart product: {url}")

        # Mock implementation - extract product ID from URL
        # Walmart URLs typically have format: /ip/Product-Name/PRODUCT_ID
        id_match = re.search(r"/ip/[^/]+/(\d+)", url)
        if not id_match:
            logger.warning(f"Could not extract product ID from URL: {url}")
            return None

        product_id = id_match.group(1)

        # Search for product in mock catalog
        for category, products in self._mock_products.items():
            for product in products:
                if product["product_id"] == product_id:
                    # Add price variation
                    price_variation = random.uniform(-0.03, 0.03)
                    current_price = product["price"] * (1 + price_variation)

                    data = {
                        "product_id": product["product_id"],
                        "product_name": product["product_name"],
                        "price": round(current_price, 2),
                        "category": self._normalize_category(category),
                        "subcategory": product["subcategory"],
                        "retailer": self.retailer_name,
                        "url": url,
                        "timestamp": datetime.utcnow(),
                        "in_stock": random.random() > 0.05,  # 95% in stock
                        "rating": round(random.uniform(3.0, 5.0), 1),
                        "review_count": random.randint(5, 5000),
                    }

                    return self.validate_price_data(data)

        logger.warning(f"Product not found in catalog: {product_id}")
        return None

    def scrape_category(self, category, max_products=100):
        """Scrape all products in a category."""
        logger.info(f"Scraping Walmart category: {category}")

        products = []
        mock_products = self._mock_products.get(category.lower(), [])

        for product in mock_products[:max_products]:
            # Simulate price variation
            price_variation = random.uniform(-0.03, 0.03)
            current_price = product["price"] * (1 + price_variation)

            # Generate Walmart-style URL
            product_slug = product["product_name"].lower().replace(" ", "-")
            url = f"{self.base_url}/ip/{product_slug}/{product['product_id']}"

            data = {
                "product_id": product["product_id"],
                "product_name": product["product_name"],
                "price": round(current_price, 2),
                "category": self._normalize_category(category),
                "subcategory": product["subcategory"],
                "retailer": self.retailer_name,
                "url": url,
                "timestamp": datetime.utcnow(),
                "in_stock": random.random() > 0.05,
            }

            validated = self.validate_price_data(data)
            if validated:
                products.append(validated)

        logger.info(f"Scraped {len(products)} products from Walmart/{category}")
        return products

    def get_categories(self):
        """Get available categories."""
        return list(self._mock_products.keys())


def create_scraper(retailer, config=None):
    """Factory function to create scrapers by retailer name."""
    scrapers = {
        "amazon": AmazonScraper,
        "walmart": WalmartScraper,
    }

    retailer_lower = retailer.lower()
    if retailer_lower not in scrapers:
        raise ValueError(
            f"Unsupported retailer: {retailer}. " f"Available: {list(scrapers.keys())}"
        )

    return scrapers[retailer_lower](config)
