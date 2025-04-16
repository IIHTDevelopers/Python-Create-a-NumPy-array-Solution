import unittest
import numpy as np
from mainclass import SalesAnalysis
from test.TestUtils import TestUtils



class BoundaryTests(unittest.TestCase):    
    def test_single_product_sales(self):
        """Test system with only one product"""
        single_product = SalesAnalysis([101], [5], [10.0])
        obj = single_product.total_revenue()
        expected_output = np.array([50.0], dtype=np.float32)
        test_obj = TestUtils()
        if np.array_equal(obj, expected_output):
            test_obj.yakshaAssert("TestSingleProductSales", True, "boundary")
            print("TestSingleProductSales = Passed")
        else:
            test_obj.yakshaAssert("TestSingleProductSales", False, "boundary")
            print("TestSingleProductSales = Failed")
