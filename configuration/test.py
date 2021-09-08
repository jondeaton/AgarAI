import unittest

import configuration

class MyTestCase(unittest.TestCase):

    def test_load_config(self):
        config = configuration.load('test_config')


if __name__ == '__main__':
    unittest.main()
