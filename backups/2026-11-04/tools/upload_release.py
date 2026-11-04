#!/usr/bin/env python3
"""Publish immutable, checksummed backup assets and re-download every asset for verification."""
import argparse, getpass, hashlib, json, os, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import requests
import tempfile

REPOSITORY='rgm-medseg/Mixing-Meets-Weighting'
TAG='comixw-critical-backup-2026-11-04'
API='https://api.github.com'
def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def token():
    for k in ['GH_TOKEN','GITHUB_TOKEN']:
        if os.environ.get(k):return os.environ[k]
    p=Path.home()/'.config/comixw-backup/github-token'
    if p.is_file():
        if p.stat().st_mode & 0o077:raise RuntimeError('Credential file must have mode 600')
        return p.read_text().strip()
    if shutil.which('gh'):
        r=subprocess.run(['gh','auth','token'],capture_output=True,text=True)
        if r.returncode==0:return r.stdout.strip()
    # Standard GitHub CLI token store, if a login was completed after backup collection began.
    p=Path.home()/'.config/gh/hosts.yml'
    if p.is_file():
        for line in p.read_text().splitlines():
            if line.strip().startswith('oauth_token:'):return line.split(':',1)[1].strip()
    return None

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--catalog',type=Path,required=True);a.add_argument('--assets',type=Path,required=True)
    a.add_argument('--download-cache',type=Path,required=True);a.add_argument('--login',action='store_true',help='Prompt privately for a GitHub API token and store it outside the backup')
    args=a.parse_args();cat=args.catalog.resolve();assets=args.assets.resolve();cache=args.download_cache.resolve();cache.mkdir(parents=True,exist_ok=True)
    receipt={'repository':REPOSITORY,'tag':TAG,'time_utc':datetime.now(timezone.utc).isoformat(),'assets':[],
             'file_verification':'NOT_RUN','paper_metrics_reproduced':False,'training_started':False,'evaluation_started':False}
    report=cat/'reports/remote_verification.json'
    def record():report.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    if args.login:
        value=getpass.getpass('GitHub token (hidden; repository Contents write permission required): ').strip()
        if not value:raise SystemExit('Empty token')
        p=Path.home()/'.config/comixw-backup/github-token';p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
        os.fchmod(fd,0o600)
        with os.fdopen(fd,'w') as f:f.write(value+'\n')
        del value
    credential=token()
    if not credential:
        receipt.update(status='BLOCKED_GITHUB_API_LOGIN',reason='SSH Git authentication succeeds, but no GitHub API credential is available. No Release was created and no asset upload was attempted.')
        record();print(receipt['reason']);return 3
    sess=requests.Session();sess.headers.update(Authorization='Bearer '+credential,Accept='application/vnd.github+json',**{'X-GitHub-Api-Version':'2022-11-28'})
    del credential
    def call(method,path,**kw):
        r=sess.request(method,API+path,timeout=(30,120),**kw)
        if r.status_code>=400:raise RuntimeError(f'GitHub API {method} {path}: HTTP {r.status_code}; {r.json().get("message", "request failed")}')
        return r.json()
    try:
        user=call('GET','/user');repo=call('GET','/repos/'+REPOSITORY)
        receipt['authenticated_user']=user['login'];receipt['repository_permissions']=repo.get('permissions',{})
        if not repo.get('permissions',{}).get('push'):raise RuntimeError('Authenticated account has no push permission to target repository')
        publication=json.loads((cat/'publication.json').read_text())
        ref=call('GET',f'/repos/{REPOSITORY}/git/ref/tags/{TAG}')
        commit=publication.get('backup_commit') or ref['object']['sha']
        remote=call('GET',f'/repos/{REPOSITORY}/commits/{commit}')
        if remote['sha']!=commit:raise RuntimeError('Backup commit missing remotely')
        receipt['backup_commit']=commit
        if ref['object']['sha']!=commit:raise RuntimeError('Existing backup tag points to a different commit; refusing to change it')
        releases=call('GET',f'/repos/{REPOSITORY}/releases?per_page=100')
        release=next((r for r in releases if r['tag_name']==TAG),None)
        if release is None:
            release=call('POST',f'/repos/{REPOSITORY}/releases',json=dict(tag_name=TAG,target_commitish=commit,
                name='CoMixW critical experiment backup — 2026-11-04',draft=True,prerelease=False,make_latest='false',
                body='Read-only backup of five paper settings and key ablations. Checksums, experiment bindings, privacy exclusions and restoration commands are committed under backups/2026-11-04/. File hashes and CPU loading are verified separately from metric reproduction. No training or evaluation was run.'))
        receipt['release_url']=release['html_url'];receipt['release_id']=release['id'];record()
        metadata=json.loads((cat/'assets.json').read_text())
        metadata.extend([{'name':'SHA256SUMS.assets','sha256':sha256(cat/'SHA256SUMS.assets'),'size':(cat/'SHA256SUMS.assets').stat().st_size,'kind':'checksums','local_catalog':True}])
        for item in metadata:
            source=(cat if item.get('local_catalog') else assets)/item['name']
            if sha256(source)!=item['sha256'] or source.stat().st_size!=item['size']:raise RuntimeError('Local asset integrity failed: '+item['name'])
            existing=call('GET',f'/repos/{REPOSITORY}/releases/{release["id"]}/assets?per_page=100')
            asset=next((x for x in existing if x['name']==item['name']),None)
            if asset is None:
                print('UPLOAD',item['name'],item['size'],flush=True)
                # No clobber or DELETE: an interrupted upload is retained and reported for review.
                with source.open('rb') as f:
                    r=sess.post(release['upload_url'].split('{',1)[0],params={'name':item['name']},data=f,
                                headers={'Content-Type':'application/octet-stream','Content-Length':str(item['size'])},timeout=(30,1800))
                if r.status_code!=201:raise RuntimeError('Upload failed for '+item['name']+': HTTP '+str(r.status_code))
                asset=r.json()
            if asset['size']!=item['size']:raise RuntimeError('Existing asset has a different size; refusing to overwrite '+item['name'])
            # Re-download using the API asset endpoint, including newly uploaded files. Do not reuse local originals.
            target=cache/item['name'];partial=target.with_name(target.name+'.partial')
            print('REMOTE DOWNLOAD',item['name'],flush=True)
            # Avoid forwarding the API bearer token to a release CDN.
            r=sess.get(asset['url'],headers={'Accept':'application/octet-stream'},stream=True,allow_redirects=False,timeout=(30,120))
            if r.is_redirect:
                location=r.headers['Location'];r.close();r=requests.get(location,stream=True,timeout=(30,120))
            r.raise_for_status()
            with partial.open('wb') as f:
                for b in r.iter_content(8*1024**2):
                    if b:f.write(b)
            r.close();actual=sha256(partial)
            if actual!=item['sha256']:raise RuntimeError('Remote download SHA mismatch for '+item['name'])
            partial.replace(target)
            receipt['assets'].append(dict(name=item['name'],url=asset['browser_download_url'],size=asset['size'],sha256=actual,remote_download_sha256='PASS'))
            record()
        if release['draft']:
            release=call('PATCH',f'/repos/{REPOSITORY}/releases/{release["id"]}',json={'draft':False,'make_latest':'false'})
        receipt.update(status='UPLOADED_AND_REMOTE_SHA256_VERIFIED',release_url=release['html_url'],file_verification='PASS')
        record();print('Remote SHA-256 verification complete:',receipt['release_url'])
        restore=cat/'tools/restore_backup.py'
        dest=Path(tempfile.mkdtemp(prefix='restore-validation-',dir=cache))
        subprocess.run([sys.executable,str(restore),'--catalog',str(cat),'--assets',str(cache),'--destination',str(dest)],check=True)
        receipt['remote_archive_restoration']='PASS';record()
        return 0
    except Exception as ex:
        receipt.update(status='INCOMPLETE',error=type(ex).__name__+': '+str(ex));record();print(receipt['error'],file=sys.stderr);return 4

if __name__=='__main__':raise SystemExit(main())
