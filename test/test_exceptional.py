import unittest
from mainclass import SalesAnalysis
from test.TestUtils import TestUtils


class ExceptionalTests(unittest.TestCase):
    def test_invalid_product_id(self):
        """Test handling of non-existing product ID"""
        test_obj = TestUtils()
        try:
            single_product = SalesAnalysis([101], [5], [10.0])
            single_product.get_product_sales(999)
            test_obj.yakshaAssert("TestInvalidProductId", False, "exceptional")
            print("TestInvalidProductId = Failed")
        except ValueError as e:
            if str(e) == "Product ID not found":
                test_obj.yakshaAssert("TestInvalidProductId", True, "exceptional")
                print("TestInvalidProductId = Passed")
            else:
                test_obj.yakshaAssert("TestInvalidProductId", False, "exceptional")
                print("TestInvalidProductId = Failed")

    def test_weekly_sales_invalid_length(self):
        """Test if ValueError is raised when reshaping non-multiple of 7"""
        invalid_sales_data = SalesAnalysis([101, 102], [2, 3, 4], [1, 2, 3])
        test_obj = TestUtils()
        try:
            invalid_sales_data.weekly_sales()
            test_obj.yakshaAssert("TestWeeklySalesInvalidLength", False, "exceptional")
            print("TestWeeklySalesInvalidLength = Failed")
        except ValueError as e:
            if str(e) == "Sales data must be in multiples of 7 for weekly reshaping":
                test_obj.yakshaAssert("TestWeeklySalesInvalidLength", True, "exceptional")
                print("TestWeeklySalesInvalidLength = Passed")
            else:
                test_obj.yakshaAssert("TestWeeklySalesInvalidLength", False, "exceptional")
                print("TestWeeklySalesInvalidLength = Failed")
