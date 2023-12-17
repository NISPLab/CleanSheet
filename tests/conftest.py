from datetime import datetime
from pathlib import Path

import pytest
from pytest import TempPathFactory


@pytest.fixture(scope='function')
def tmp_work_dirs(tmp_path_factory: TempPathFactory) -> Path:
    relative_work_dirs = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    work_dirs = tmp_path_factory.mktemp(relative_work_dirs)

    return work_dirs
