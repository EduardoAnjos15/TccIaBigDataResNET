from pathlib import Path
import os
import runpy
import sys

if len(sys.argv) < 4:
    raise SystemExit("usage: run_render.py RENDERER INPUT OUTPUT_DIR [extra args]")

renderer = Path(sys.argv[1])
input_path = sys.argv[2]
output_dir = sys.argv[3]
extra = sys.argv[4:]
os.environ["PATH"] = r"C:\Program Files\LibreOffice\program;" + os.environ.get("PATH", "")
sys.argv = [str(renderer), input_path, "--output_dir", output_dir, *extra]
runpy.run_path(str(renderer), run_name="__main__")
