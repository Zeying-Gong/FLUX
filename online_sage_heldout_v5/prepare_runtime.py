"""Link external licensed runtime bundles; no download of assets or weights."""
import argparse,pathlib
p=argparse.ArgumentParser();p.add_argument("--runtime-source",type=pathlib.Path,required=True)
a=p.parse_args();root=pathlib.Path(__file__).resolve().parent
for name in ["runtime_deps","runtime_exts"]:
 source=(a.runtime_source/name).resolve();target=root/name
 if not source.is_dir():raise FileNotFoundError(source)
 if target.exists() or target.is_symlink():
  if target.resolve()!=source:raise RuntimeError("Refusing to replace "+str(target))
 else:target.symlink_to(source,target_is_directory=True)
 print(name,"->",source)
