import unittest
import numpy as np
from mainclass import SalesAnalysis
from test.TestUtils import TestUtils


class FunctionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        cls.product_ids = [101, 102, 103, 104, 105]
        cls.units_sold = [20, 15, 50, 30, 10]
        cls.prices = [10.5, 20.0, 5.0, 15.0, 8.0]
        cls.sales_analysis = SalesAnalysis(cls.product_ids, cls.units_sold, cls.prices)

    def test_total_revenue(self):
        """Test if total revenue is correctly calculated"""
        obj = self.sales_analysis.total_revenue()
        expected_revenue = np.array([210.0, 300.0, 250.0, 450.0, 80.0], dtype=np.float32)
        test_obj = TestUtils()
        if np.array_equal(obj, expected_revenue):
            test_obj.yakshaAssert("TestTotalRevenue", True, "functional")
            print("TestTotalRevenue = Passed")
        else:
            test_obj.yakshaAssert("TestTotalRevenue", False, "functional")
            print("TestTotalRevenue = Failed")


    def test_normalized_sales(self):
        """Test if sales data is correctly normalized"""
        normalized_data = self.sales_analysis.normalize_sales()
        expected_normalized = np.array([0.25, 0.125, 1.0, 0.5, 0.0], dtype=np.float32)

        test_obj = TestUtils()
        if np.array_equal(normalized_data, expected_normalized):
            test_obj.yakshaAssert("TestNormalizedSales", True, "functional")
            print("TestNormalizedSales = Passed")
        else:
            test_obj.yakshaAssert("TestNormalizedSales", False, "functional")
            print("TestNormalizedSales = Failed")