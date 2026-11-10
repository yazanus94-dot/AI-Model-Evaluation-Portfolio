import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from vllm_eval.cli import main  # noqa: E402

main()
