import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from sglang_judge.cli import main  # noqa: E402

main()
