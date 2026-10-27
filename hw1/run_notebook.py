import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
notebook = nbformat.read(ROOT / 'experiments.ipynb', as_version=4)
manager = KernelManager(kernel_name='python3')
manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']

def progress(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Executing code cell {cell_index}: {cell.source.splitlines()[0][:100]}', flush=True)

client = NotebookClient(notebook, km=manager, timeout=3600, resources={'metadata': {'path': str(ROOT)}},
                        on_cell_start=progress)
try:
    client.execute()
finally:
    nbformat.write(notebook, ROOT / 'experiments.ipynb')
print('Saved executed experiments.ipynb', flush=True)
