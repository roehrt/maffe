"""Run against an installed wheel: python tests/test_python.py."""

from importlib.metadata import version
import unittest

import maffe


class PythonBindingsTest(unittest.TestCase):
    first = "((a,b),(c,d));"
    second = "((a,c),(b,d));"

    def assert_partition(self, partition, size):
        self.assertEqual(len(partition), size)
        self.assertTrue(all(partition))
        self.assertEqual(sorted(leaf for block in partition for leaf in block), list("abcd"))

    def test_version(self):
        self.assertEqual(maffe.__version__, version("maffe"))

    def test_rooted_identical(self):
        self.assert_partition(maffe.solve([self.first, self.first]), 1)

    def test_rooted_disagreement(self):
        trees = [self.first, self.second]
        partition = maffe.solve(trees)
        self.assert_partition(partition, 3)
        forest = maffe.restrict_solution(trees, partition)
        self.assertEqual(len(forest), len(partition))
        self.assertTrue(all(tree.endswith(";") for tree in forest))

    def test_unrooted_identical(self):
        self.assert_partition(maffe.solve_unrooted([self.first, self.first]), 1)

    def test_unrooted_disagreement(self):
        self.assert_partition(maffe.solve_unrooted([self.first, self.second]), 2)

    def test_distance(self):
        self.assertEqual(maffe.rspr_dist(self.first, self.first), 0)
        self.assertEqual(maffe.rspr_dist("((a,b),c);", "((a,c),b);"), 1)

    def test_restriction(self):
        self.assertEqual(maffe.restrict_tree([self.first], ["a", "b"]), "(a,b);")
        self.assertEqual(maffe.restrict_tree([self.first], ["c"]), "c;")
        self.assertEqual(
            maffe.restrict_solution([self.first], [["a", "b"], ["c", "d"]]),
            ["(a,b);", "(c,d);"],
        )

    def test_log_levels(self):
        for level in (0, "quiet", maffe.LogLevel.QUIET):
            with self.subTest(level=level):
                self.assert_partition(maffe.solve([self.first, self.first], log_level=level), 1)
        for level, error in ((True, TypeError), (1.5, TypeError), ("invalid", ValueError), (3, ValueError)):
            with self.subTest(level=level), self.assertRaises(error):
                maffe.solve([self.first, self.first], log_level=level)

    def test_invalid_trees(self):
        for trees in ([], ["((a,a),b);"], ["(a,b,c);"], [self.first, "((a,b),(c,e));"]):
            with self.subTest(trees=trees), self.assertRaises(ValueError):
                maffe.solve(trees)
        with self.assertRaises(ValueError):
            maffe.solve_unrooted([self.first])
        with self.assertRaises(ValueError):
            maffe.restrict_tree([self.first], ["unknown"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
