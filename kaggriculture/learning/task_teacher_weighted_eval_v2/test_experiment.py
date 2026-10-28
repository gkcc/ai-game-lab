import unittest
from .experiment import validate_new_seeds


class SeedTests(unittest.TestCase):
    def test_rejects_failed_reserved_and_recognized_seeds(self):
        validate_new_seeds([11,12,13],{1},[2,3,4,5,6,7],[8,9,10])
        for seed in (1,2,7,8,10):
            with self.subTest(seed=seed),self.assertRaises(ValueError):
                validate_new_seeds([seed,12,13],{1},[2,3,4,5,6,7],[8,9,10])
        with self.assertRaises(ValueError): validate_new_seeds([11,11,13],set(),[],[])


if __name__=='__main__': unittest.main()
