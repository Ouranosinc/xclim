from xclim import build_module
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
        self.assertRaises(Warning, build_module, name, objs, mode='warn')


if __name__ == '__main__':
    unittest.main()
