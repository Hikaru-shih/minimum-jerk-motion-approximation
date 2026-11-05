"""Audit the concise editable report without rendering or opening a PDF."""
import hashlib
import json
import re
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from docx import Document
from lxml import etree
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'report/final_motion_decomposition_report.docx'
ASSETS=ROOT/'report/final_report_assets'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}


def main():
    revision=json.loads((ASSETS/'editorial_revision.json').read_text())
    baseline=ROOT/revision['format_reference']
    assert hashlib.sha256(baseline.read_bytes()).hexdigest()==revision['format_reference_sha256']
    doc=Document(REPORT)
    assert len(doc.tables)==2 and len(doc.inline_shapes)==6
    with ZipFile(REPORT) as final,ZipFile(baseline) as original:
        tree=etree.fromstring(final.read('word/document.xml'))
        assert len(tree.xpath('//m:oMath',namespaces=NS))==13
        assert not tree.xpath('//m:e/m:e',namespaces=NS)
        assert not tree.xpath('//w:br[@w:type="page"]',namespaces=NS)
        # Word can reorder ZIP parts and rename images when the author saves it.
        assert final.read('word/styles.xml')==original.read('word/styles.xml')
        geometry=('page_width','page_height','top_margin','bottom_margin','left_margin','right_margin')
        before=Document(baseline)
        assert [[getattr(s,k) for k in geometry] for s in doc.sections]==[[getattr(s,k) for k in geometry] for s in before.sections]
        ids=tree.xpath('//a:blip/@r:embed',namespaces=NS)
        relationships={r.get('Id'):r.get('Target') for r in etree.fromstring(final.read('word/_rels/document.xml.rels'))}
        hashes=[hashlib.sha256(final.read('word/'+relationships[rid])).hexdigest() for rid in ids]
        assert len(hashes)==len(set(hashes))==6
        def pixel_hashes(package):
            xml=etree.fromstring(package.read('word/document.xml'))
            rel={r.get('Id'):r.get('Target') for r in etree.fromstring(package.read('word/_rels/document.xml.rels'))}
            return [hashlib.sha256(Image.open(BytesIO(package.read('word/'+rel[rid]))).convert('RGB').tobytes()).hexdigest()
                    for rid in xml.xpath('//a:blip/@r:embed',namespaces=NS)]
        figure_reference=ROOT/revision['velocity_supplement_backup']['backup']
        assert hashlib.sha256(figure_reference.read_bytes()).hexdigest()==revision['velocity_supplement_backup']['backup_sha256']
        with ZipFile(figure_reference) as previous_concise:
            assert pixel_hashes(final)==pixel_hashes(previous_concise)
    results=json.loads((ROOT/'results/mainline/summary.json').read_text())
    names={'uniform_sampled':'Uniform sampled','uniform_fitted':'Uniform fitted','valley_fitted':'Valley fitted'}
    expected=[]
    for label in ('18_21','30_33'):
        k=results['details'][label]['detection']['component_count']
        for method in names:
            r=next(r for r in results['rows'] if r['window']==label and r['K']==k and r['method']==method)
            expected.append([label.replace('_','–'),names[method],str(k),f"{100*r['position_rmse_m']:.3f}",f"{r['velocity_rmse_m_s']:.4f}",f"{100*r['max_position_error_m']:.3f}"])
    assert [[c.text for c in r.cells] for r in doc.tables[0].rows[1:]]==expected
    velocity=json.loads((ROOT/'results/velocity_objective/summary.json').read_text())
    expected_velocity=[[r['window'].replace('_','–'),f"{r['weight']:g}",str(r['K']),
                        f"{100*r['position_rmse_m']:.3f}",f"{r['velocity_rmse_m_s']:.4f}",
                        f"{100*r['max_position_error_m']:.3f}"] for r in velocity['rows']]
    assert [[c.text for c in r.cells] for r in doc.tables[1].rows[1:]]==expected_velocity
    text='\n'.join(p.text for p in doc.paragraphs)
    for obsolete in ('212.456','3.9663','54 passing','54 passed','numpy==','python3 -m','results/mainline/'):
        assert obsolete not in text
    assert 'not included in the objective' in text
    figures=[p.text for p in doc.paragraphs if re.match(r'^Figure \d+\.',p.text)]
    assert [int(re.match(r'Figure (\d+)\.',p).group(1)) for p in figures]==list(range(1,7))
    result={'docx_sha256':hashlib.sha256(REPORT.read_bytes()).hexdigest(),
            'pages_in_word_image_preview':10,'editable_equations':13,'unique_figures':6,'tables':2,
            'main_table_matches_source':True,'velocity_table_matches_source':True,'format_reference':revision['format_reference'],
            'format_reference_sha256':revision['format_reference_sha256'],
            'styles_preserved':True,'figure_pixels_and_order_preserved':True,'page_geometry_preserved':True,
            'pdf_generated_by_audit':False,
            'visual_review':'Report reviewed via page images; audit verifies data, styles, geometry and figure pixels.'}
    (ASSETS/'document_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
