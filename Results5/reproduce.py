from pathlib import Path
import os
import subprocess
import sys
import tempfile

code = Path(__file__).resolve().parent / "code"
with tempfile.TemporaryDirectory(prefix="fig5_render_") as work:
    env = dict(os.environ, FIG5_WORK=work, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    for script in ["heat.py", "assemble.py", "finalize.py"]:
        subprocess.run([sys.executable, str(code/script)], env=env, check=True)
