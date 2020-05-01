import unittest
from writedata import Value, get_rows

test_data = [Value('2020-04-15', 'Москва', 1355), Value('2020-04-15', 'Волгоградская область', 6),
             Value('2020-04-13', 'Москва', 1489), Value('2020-04-13', 'Московская область', 460)]


class WritedataTest(unittest.TestCase):
    def test_get_rows(self):
        expected = [
            ['', '2020-04-13', '2020-04-15'],
            ['Волгоградская область', 0, 6],
            ['Москва', 1489, 1355],
            ['Московская область', 460, 0]
        ]
        got = get_rows(test_data)
        self.assertEqual(got, expected)
