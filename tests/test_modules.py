import sys

from xclim import build_module

if sys.version_info < (3, 0):
    import unittest2 as unittest
else:
    import unittest


class TestBuildModules(unittest.TestCase):

    def test_nonexistent_process_build_failure(self):
        name = ""
        objs = {'k1': 'v1', 'k2': 'v2'}
        self.assertRaises(AttributeError, build_module, name, objs, mode='warn')

    def test_quiet_build_failure(self):
        name = None
        objs = {}
        self.assertRaises(TypeError, build_module, name, objs, mode='ignore')

    def test_loud_build_failure(self):
        name = ""
        objs = {'k1': None, 'k2': None}
        self.assertWarns(Warning, build_module, name, objs, mode='warn')

    def test_raise_build_failure(self):
        name = ""
        objs = {'k1': None, 'k2': None}
        self.assertRaises(NotImplementedError, build_module, name, objs, mode='raise')


if __name__ == '__main__':
    unittest.main()
