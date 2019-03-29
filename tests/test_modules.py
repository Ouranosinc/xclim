import pytest
from xclim import build_module


class TestBuildModules():

    def test_nonexistent_process_build_failure(self):
        name = ""
        objs = {'k1': 'v1', 'k2': 'v2'}
        with pytest.raises(AttributeError):
            build_module(name, objs, mode='warn')

    def test_quiet_build_failure(self):
        name = None
        objs = {}
        with pytest.raises(TypeError):
            build_module(name, objs, mode='ignore')

    def test_loud_build_failure(self):
        name = ""
        objs = {'k1': None, 'k2': None}
        with pytest.warns(Warning):
            build_module(name, objs, mode='warn')

    def test_raise_build_failure(self):
        name = ""
        objs = {'k1': None, 'k2': None}
        with pytest.raises(NotImplementedError):
            build_module(name, objs, mode='raise')


class TestICCLIM():

    def test_exists(self):
        from xclim import icclim
        assert getattr(icclim, 'TG', None) is not None
