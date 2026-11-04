#!/usr/bin/env python3
"""Download, verify, and materialize this backup. Standard library only."""
import argparse, gzip, hashlib, json, os, shutil, tarfile, urllib.request
from pathlib import Path

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def safe_path(root,relative):
    p=(root/relative).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('Unsafe path: '+relative)
    return p

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--catalog',type=Path,default=Path(__file__).resolve().parents[1])
    a.add_argument('--assets',type=Path,default=Path('assets'))
    a.add_argument('--download',action='store_true')
    a.add_argument('--verify-only',action='store_true')
    a.add_argument('--destination',type=Path)
    args=a.parse_args();cat=args.catalog.resolve();assets=args.assets.resolve();assets.mkdir(parents=True,exist_ok=True)
    meta=json.loads((cat/'assets.json').read_text())
    for asset in meta:
        p=assets/asset['name']
        if not p.exists() and args.download:
            temp=p.with_suffix(p.suffix+'.partial')
            print('Downloading',asset['name'],flush=True)
            req=urllib.request.Request(asset['download_url'],headers={'User-Agent':'CoMixW-backup-restore'})
            with urllib.request.urlopen(req,timeout=120) as response,temp.open('wb') as out:shutil.copyfileobj(response,out,8*1024**2)
            if sha256(temp)!=asset['sha256']:raise ValueError('Downloaded SHA mismatch: '+asset['name'])
            temp.replace(p)
        if not p.exists():raise FileNotFoundError(p)
        if p.stat().st_size!=asset['size'] or sha256(p)!=asset['sha256']:raise ValueError('Asset SHA mismatch: '+asset['name'])
        print('SHA256 PASS',asset['name'],flush=True)
    if args.verify_only:return
    if args.destination is None:raise SystemExit('Specify --destination or --verify-only')
    dest=args.destination.resolve()
    if dest.exists() and any(dest.iterdir()):raise SystemExit('Destination must be empty; original files are never overwritten')
    dest.mkdir(parents=True,exist_ok=True);store=dest/'object_store';store.mkdir()
    # Current bundles fit individually below 2 GiB. Multipart support keeps lexical part order explicit in assets.json.
    archives={}
    for asset in meta:
        if asset['kind']!='weights':continue
        archives.setdefault(asset.get('archive_name',asset['name']),[]).append(asset)
    for archive,parts in archives.items():
        parts=sorted(parts,key=lambda x:x.get('part',0))
        source=assets/parts[0]['name']
        if len(parts)>1:
            source=assets/archive
            with source.open('xb') as out:
                for part in parts:
                    with (assets/part['name']).open('rb') as f:shutil.copyfileobj(f,out,8*1024**2)
        with tarfile.open(source,'r:gz') as tf:
            for member in tf:
                if not member.isfile() or not member.name.startswith('objects/'):raise ValueError('Unexpected archive member')
                target=safe_path(store,member.name);target.parent.mkdir(parents=True,exist_ok=True)
                with tf.extractfile(member) as f,target.open('xb') as out:shutil.copyfileobj(f,out,8*1024**2)
        if len(parts)>1:source.unlink()
    weights=json.loads((cat/'weights.json').read_text())
    for w in weights:
        src=store/w['object']
        if sha256(src)!=w['sha256']:raise ValueError('Weight object SHA mismatch: '+w['object'])
        target=safe_path(dest/'project',w['source']);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha256(target)!=w['sha256']:raise ValueError('Conflicting weight reference')
        else:os.link(src,target)
    for f in json.loads((cat/'files.json').read_text()):
        if not f['backup_path'].startswith('records/'):continue
        src=cat/f['backup_path']
        if sha256(src)!=f['sha256']:raise ValueError('Catalog SHA mismatch: '+f['backup_path'])
        source=f.get('source','')
        if source.startswith('publication/'):target=safe_path(dest/'publication',source.removeprefix('publication/'))
        else:target=safe_path(dest/'project',source)
        target.parent.mkdir(parents=True,exist_ok=True)
        opener=gzip.open if f.get('compression')=='gzip' else open
        with opener(src,'rb') as inp,target.open('xb') as out:shutil.copyfileobj(inp,out,8*1024**2)
        if sha256(target)!=f.get('content_sha256',f['sha256']):raise ValueError('Restored content SHA mismatch')
    print('RESTORE PASS',dest)
    print('Weights are hard-linked to object_store; keep that directory until restoration is complete.')

if __name__=='__main__':main()
