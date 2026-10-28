import unittest
from .experiment import validate_seed_roles


class SeedRoleTests(unittest.TestCase):
    def test_development_and_reserved_seeds_must_be_distinct_and_new(self):
        validate_seed_roles([1, 2], [3, 4, 5, 6], {7, 8})
        for train, reserved, excluded in (
            ([1, 1], [3, 4, 5, 6], set()),
            ([1, 2], [2, 4, 5, 6], set()),
            ([1, 2], [3, 4, 5, 6], {6}),
            ([True, 2], [3, 4, 5, 6], set()),
            ([1, 2], [3, 4, 5, 2**32], set()),
            ([1, 2], [3, 4], set()),
        ):
            with self.subTest(train=train, reserved=reserved, excluded=excluded):
                with self.assertRaises(ValueError):
                    validate_seed_roles(train, reserved, excluded)


if __name__ == '__main__':
    unittest.main()
