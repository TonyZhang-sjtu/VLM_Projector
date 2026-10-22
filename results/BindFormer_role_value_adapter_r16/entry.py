"""Explicit isolated dispatcher; propagates extension to training/evaluation children."""
import importlib
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
ALLOWED={'src.cli.baseline_sweep','src.cli.train','src.cli.evaluate_checkpoint',
         'src.cli.evaluate','src.cli.score_reports'}

def rewrite(command):
    if not isinstance(command,(list,tuple)) or '-m' not in command:return command
    i=command.index('-m')
    if i+1<len(command) and command[i+1] in ALLOWED and Path(command[0]).resolve()==Path(sys.executable).resolve():
        return [*command[:i],str(HERE/'entry.py'),command[i+1],*command[i+2:]]
    return command

def main():
    if len(sys.argv)<2 or sys.argv[1] not in ALLOWED:
        raise SystemExit('Usage: entry.py src.cli.{baseline_sweep,train,evaluate_checkpoint,evaluate,score_reports} ...')
    target=sys.argv[1];sys.argv=[target,*sys.argv[2:]]
    # Text-only clinical scoring does not load the experimental projector.
    # Its CLI must activate the offline dependencies before Torch/NumPy import.
    if target=='src.cli.score_reports':
        importlib.import_module(target).main()
        return
    from runtime import install
    install()
    original=subprocess.Popen
    class LocalPopen(original):
        def __init__(self,args,*a,**kw):super().__init__(rewrite(args),*a,**kw)
    subprocess.Popen=LocalPopen
    module=importlib.import_module(target)
    if target=='src.cli.baseline_sweep':
        old_build=module.build_plan
        def build(args):
            plan=old_build(args)
            for run in plan['runs']:
                for field in ('train_command','eval_command'):run[field]=rewrite(run[field])
            return plan
        module.build_plan=build
    module.main()

if __name__=='__main__':main()
