import numpy as np
import pandas as pd

class SalesAnalysis:
    def __init__(self, product_ids, units_sold, prices):
        """Initialize NumPy arrays for product sales analysis"""
        self.product_ids = np.array(product_ids, dtype=np.int32)
        self.units_sold = np.array(units_sold, dtype=np.float32)
        self.prices = np.array(prices, dtype=np.float32)

    def total_revenue(self):
        """Calculate total revenue per product per day"""
        return self.units_sold * self.prices

    def top_selling_products(self, top_n=5):
        """Return the product IDs of top N selling products"""
        sorted_indices = np.argsort(self.units_sold)[-top_n:][::-1]  # Sort descending
        return self.product_ids[sorted_indices]

    def weekly_sales(self):
        """Reshape daily sales into weekly format (7 days per week)"""
        if len(self.units_sold) % 7 != 0:
            raise ValueError("Sales data must be in multiples of 7 for weekly reshaping")
        return self.units_sold.reshape(-1, 7)

    def get_product_sales(self, product_id):
        """Retrieve sales for a specific product"""
        if product_id not in self.product_ids:
            raise ValueError("Product ID not found")
        index = np.where(self.product_ids == product_id)[0][0]
        return self.units_sold[index]


    def normalize_sales(self):
        """Normalize the sales data between 0 and 1"""
        min_sales = np.min(self.units_sold)
        max_sales = np.max(self.units_sold)

        normalized_sales = (self.units_sold - min_sales) / (max_sales - min_sales)
        return normalized_sales


    def apply_normalize_sales(self):
        """Normalize the units_sold data using the apply() method"""
        return np.array([self.apply_function(x) for x in self.units_sold])

    def apply_function(self, x):
        """Helper function for normalization using apply()"""
        min_sales = np.min(self.units_sold)
        max_sales = np.max(self.units_sold)
        return (x - min_sales) / (max_sales - min_sales)
