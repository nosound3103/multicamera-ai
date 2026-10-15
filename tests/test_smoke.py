from importlib.metadata import version

import multicamera_ai


def test_package_import():
    assert multicamera_ai.__name__ == "multicamera_ai"


def test_package_version():
    assert version("multicamera-ai") == "0.1.0"
