"""Isaac 앱을 먼저 띄운 뒤 jepa_loco 단위 테스트 전체를 실행한다.

일부 테스트는 isaaclab(configclass, mdp)을 import 하는데, 이는 pxr 이 필요해 앱 없이 import 할 수 없다.

    python scripts/jepa_loco/run_tests_in_app.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli, pytest_args = parser.parse_known_args()
simulation_app = AppLauncher(args_cli).app

import pytest  # noqa: E402

TESTS = os.path.join(os.path.dirname(__file__), "..", "..", "source/unitree_rl_lab/unitree_rl_lab/jepa_loco/tests")
code = pytest.main([os.path.abspath(TESTS), "-q", "-p", "no:cacheprovider", *pytest_args])
print(f"PYTEST_EXIT_CODE={int(code)}", flush=True)  # close() 가 stdout 버퍼를 버리므로 flush
simulation_app.close()
sys.exit(int(code))
