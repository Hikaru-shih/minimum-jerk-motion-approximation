"""Check existing repository Markdown, including moved-but-not-yet-staged files."""
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    broken=[];files=0;links=0;largest=[]
    for name in sorted(set(names)):
        path=ROOT/name
        if not path.is_file():continue
        largest.append((path.stat().st_size,name))
        if path.suffix!='.md':continue
        files+=1
        for url in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if re.match(r'https?://|mailto:|#',url):continue
            links+=1
            if not (path.parent/url.split('#')[0]).exists():broken.append((name,url))
    print('Markdown files:',files,'Local links:',links,'Broken:',broken)
    large=[(n,round(size/1024/1024,2)) for size,n in sorted(largest,reverse=True) if size>=100*1024*1024]
    print('Git candidates at or above 100 MiB:',large)
    print('Total existing Git candidate size MiB:',round(sum(size for size,_ in largest)/1024/1024,2))
    assert not broken
    assert not large


if __name__=='__main__':main()
