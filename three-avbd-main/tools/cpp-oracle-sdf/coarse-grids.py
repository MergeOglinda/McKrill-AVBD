#!/usr/bin/env python3
"""Export or verify pinned upstream voxel grids, separately from the phase-0 fixtures."""
import argparse, gzip, hashlib, json, os, pathlib, subprocess, tempfile
ROOT = pathlib.Path(__file__).resolve().parents[2]
REV = '1dcdf1e7fa07d16fda15826ad8c2e6697075a76e'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode', choices=['generate', 'verify'])
parser.add_argument('--source', type=pathlib.Path, required=True)
args = parser.parse_args()
source = args.source.resolve()
def run(*cmd): return subprocess.check_output(cmd, text=True)
def digest(data): return hashlib.sha256(data).hexdigest()
if run('git', '-C', str(source), 'rev-parse', 'HEAD').strip() != REV or run('git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all').strip():
    raise SystemExit('Need clean pinned oracle checkout')
compiler = os.environ.get('CXX', 'c++')
flags = ['-O2', '-std=c++17', '-ffp-contract=off', '-fno-fast-math']
files = [source/'source'/n for n in ['shapes.cpp', 'voxel.cpp', 'mesh.cpp']]
driver = ROOT/'tools/cpp-oracle-sdf/export-coarse-grid.cpp'
manifest = dict(revision=REV, compiler=run(compiler, '--version').splitlines()[0], flags=flags, format='little-endian float32 phi,gx,gy,gz; x-fastest', driverSha256=digest(driver.read_bytes()), inputs={str(p.relative_to(source)):digest(p.read_bytes()) for p in [*files,*sorted(source.joinpath('source').glob('*.h')),*sorted(source.joinpath('assets').glob('*.obj'))]}, grids={})
destination = ROOT/'tests/fixtures/oracle-sdf-coarse-grids'
with tempfile.TemporaryDirectory(prefix='sdf-grids-') as tmp:
    executable = str(pathlib.Path(tmp)/'export')
    subprocess.run([compiler,*flags,'-I'+str(source/'source'),str(driver),*[str(p) for p in files],'-o',executable],check=True)
    for name in ['bunny','gear','cow','chair']:
        print(args.mode, name, flush=True)
        raw = pathlib.Path(tmp)/(name+'.bin')
        run(executable,str(source/'assets'),name,str(raw))
        data = raw.read_bytes()
        manifest['grids'][name] = dict(bytes=len(data),sha256=digest(data))
        path = destination/(name+'.bin.gz')
        if args.mode == 'generate':
            destination.mkdir(parents=True,exist_ok=True)
            path.write_bytes(gzip.compress(data,compresslevel=6,mtime=0))
        elif gzip.decompress(path.read_bytes()) != data:
            raise SystemExit('Grid differs: '+name)
    path = destination/'manifest.json'
    if args.mode == 'generate': path.write_text(json.dumps(manifest,indent=2)+'\n')
    elif json.loads(path.read_text()) != manifest: raise SystemExit('Grid provenance differs')
print(args.mode+' succeeded')
