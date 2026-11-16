#!/usr/bin/env python3
"""Build the pinned oracle; generate fixtures explicitly, or verify without overwriting."""
import argparse, hashlib, json, os, pathlib, subprocess, tempfile
ROOT = pathlib.Path(__file__).resolve().parents[2]
REV = '1dcdf1e7fa07d16fda15826ad8c2e6697075a76e'
SHAPES = ['box', 'sphere', 'capsule', 'cylinder', 'cone', 'torus', 'spike-plate', 'spiky-wheel', 'bend', 'twist', 'displace', 'bunny', 'gear', 'cow', 'chair']
SHAPES += ['bend-0.15', 'bend-0.55', 'bend-0.95', 'displace-0.05', 'displace-0.10', 'displace-0.20', 'twist-0.50', 'twist-2.50', 'twist-5.00']
SOURCES = ['shapes.cpp', 'mesh.cpp', 'voxel.cpp', 'collide.cpp', 'optimize.cpp', 'solver.cpp', 'scenes.cpp']
FLAGS = ['-O2', '-std=c++17', '-ffp-contract=off', '-fno-fast-math']
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode', choices=['generate', 'verify'])
parser.add_argument('--source', type=pathlib.Path, required=True, help='clean checkout of savant117/sdf-dcd-demo at the pinned revision')
args = parser.parse_args()
source = args.source.resolve()
def run(*cmd):
    return subprocess.check_output(cmd, text=True)
if run('git', '-C', str(source), 'rev-parse', 'HEAD').strip() != REV:
    raise SystemExit('Wrong oracle revision; refusing to generate or verify')
if run('git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all').strip():
    raise SystemExit('Oracle checkout is dirty; refusing to generate or verify')
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
inputs = sorted([*source.joinpath('source').glob('*.h'), *[source/'source'/f for f in SOURCES], *source.joinpath('assets').glob('*.obj')])
compiler = os.environ.get('CXX', 'c++')
manifest = dict(schemaVersion=1, repository='https://github.com/savant117/sdf-dcd-demo', revision=REV, precision='upstream float32', compiler=run(compiler,'--version').splitlines()[0], flags=FLAGS, inputs={str(p.relative_to(source)): digest(p) for p in inputs}, driverSha256=digest(ROOT/'tools/cpp-oracle-sdf/oracle.cpp'), adapters={'SDL.h': digest(ROOT/'tools/cpp-oracle-sdf/SDL.h')})
fixtures = ROOT/'tests/fixtures/oracle-sdf'
with tempfile.TemporaryDirectory(prefix='avbd-sdf-oracle-') as temp:
    executable = str(pathlib.Path(temp)/'oracle')
    subprocess.run([compiler,*FLAGS,'-I'+str(ROOT/'tools/cpp-oracle-sdf'),'-I'+str(source/'source'),str(ROOT/'tools/cpp-oracle-sdf/oracle.cpp'),*[str(source/'source'/f) for f in SOURCES],'-o',executable],check=True)
    os.environ['SDF_ORACLE_ROOT'] = str(source) + '/'
    outputs = {}
    for name in SHAPES:
        print('Sampling', name, flush=True)
        outputs[name+'.json'] = json.loads(run(executable,'sample',name,str(source/'assets')))
    outputs['optimizer.json'] = json.loads(run(executable,'optimize'))
    # Repeat the optimizer to detect accidental nondeterminism without relying on fixtures.
    if outputs['optimizer.json'] != json.loads(run(executable,'optimize')):
        raise SystemExit('Optimizer output is not repeatable')
    outputs['optimizers.json'] = json.loads(run(executable,'optimizers'))
    if outputs['optimizers.json'] != json.loads(run(executable,'optimizers')):
        raise SystemExit('Ellipsoid/gradient-descent output is not repeatable')
    for name in SHAPES:
        print('Finalizing seeds', name, flush=True)
        outputs['seeds-'+name+'.json'] = json.loads(run(executable,'seeds',name,str(source/'assets')))
    outputs['manifolds.json'] = json.loads(run(executable,'manifolds',str(source/'assets')))
    if outputs['manifolds.json'] != json.loads(run(executable,'manifolds',str(source/'assets'))):
        raise SystemExit('Manifold output is not repeatable')
    outputs['manifolds-optimizers.json'] = json.loads(run(executable,'manifolds-optimizers',str(source/'assets')))
    if outputs['manifolds-optimizers.json'] != json.loads(run(executable,'manifolds-optimizers',str(source/'assets'))):
        raise SystemExit('Optimizer manifold output is not repeatable')
    for index in range(11):
        print('Simulating scene', index, flush=True)
        outputs[f'scene-{index}.json'] = json.loads(run(executable,'scene',str(index),'360'))
    for index, values in [(3, ['0.15', '0.55', '0.95']), (4, ['0.50', '2.50', '5.00']), (5, ['0.05', '0.10', '0.20'])]:
        for k in values:
            print('Simulating warp scene', index, k, flush=True)
            outputs[f'scene-{index}-k{k}.json'] = json.loads(run(executable,'scene',str(index),'360',k))
    manifest['fixtures'] = {name: hashlib.sha256((json.dumps(value,indent=2,allow_nan=False)+'\n').encode()).hexdigest() for name,value in outputs.items()}
    outputs['manifest.json'] = manifest
    for name,value in outputs.items():
        path=fixtures/name
        if args.mode=='generate':
            fixtures.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        elif json.loads(path.read_text()) != value:
            raise SystemExit('Fixture differs: '+name+' (not overwritten)')
print(args.mode+' succeeded; '+run(compiler,'--version').splitlines()[0])
