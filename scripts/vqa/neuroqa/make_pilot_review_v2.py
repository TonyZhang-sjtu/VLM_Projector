#!/usr/bin/env python3
"""Create contact sheets for the 32 chosen pilot transforms, without approval."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from PIL import Image, ImageDraw
from src.config import file_hash
from src.data.neuroqa_manifest import write_json


def main():
    plan=ROOT/'results/vqa/neuroqa/preprocessing_v2_plan'
    summary_path=plan/'pilot_summary.json'
    summary=json.loads(summary_path.read_text())
    if any(summary['candidate_summaries'][c]['measured_cases'] != 32 for c in ('B','C','D')):
        raise RuntimeError('Complete pilot summary required for review')
    entries=[]
    for row in summary['cases']:
        c,d=row['candidates']['C'],row['candidates']['D']
        choice='C' if not c['issues'] else ('D' if not d['issues'] else None)
        if choice is None:
            raise RuntimeError(f'Neither candidate passes pilot gates: {row["case_id"]}')
        path=plan/'pilot'/choice/row['case_id']/'qc.png'
        entries.append({'case_id':row['case_id'],'candidate':choice,'image':str(path)})
    folder=plan/'pilot_visual_review'
    folder.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(entries),8):
        sheet=Image.new('RGB',(1100,2280),'white')
        draw=ImageDraw.Draw(sheet)
        for i,entry in enumerate(entries[start:start+8]):
            x=(i%2)*550;y=(i//2)*570
            draw.text((x+3,y+3),f'{entry["case_id"]} candidate={entry["candidate"]}',fill='black')
            with Image.open(entry['image']) as im:
                sheet.paste(im.convert('RGB').resize((540,540)),(x+5,y+22))
        sheet.save(folder/f'page_{start//8+1:02d}.png')
    write_json(folder/'queue.json',{'summary_sha256':file_hash(summary_path),'cases':entries,
        'scope':'three_central_planes_engineering_review_not_clinical_certification'})
    print(json.dumps({'cases':len(entries),'pages':(len(entries)+7)//8,'queue':str(folder/'queue.json')}))


if __name__=='__main__':
    main()
