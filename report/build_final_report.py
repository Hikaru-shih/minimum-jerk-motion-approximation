"""Build the editable comprehensive third-stage report from frozen evidence."""
import csv
import hashlib
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
ASSETS=HERE/'final_report_assets'
TARGET=HERE/'final_motion_decomposition_report.docx'


def element(tag,*children,**attributes):
    node=OxmlElement('m:'+tag)
    for key,value in attributes.items():node.set(qn('m:'+key),str(value))
    for child in children:
        child=run(child) if isinstance(child,str) else child
        if tag in ('e','num','den','sub','sup','oMath') and child.tag==qn('m:e'):
            for item in list(child):node.append(item)
        else:node.append(child)
    return node


def run(text):
    r=element('r');t=element('t');t.text=text;r.append(t);return r


def seq(*items):return element('e',*items)
def sub(base,index):return element('sSub',element('e',base),element('sub',index))
def sup(base,power):return element('sSup',element('e',base),element('sup',power))
def frac(top,bottom):return element('f',element('num',top),element('den',bottom))
def hat(base):return element('acc',element('accPr',element('chr',val='̂')),element('e',base))
def total(index,upper,body):
    return element('nary',element('naryPr',element('chr',val='∑'),element('limLoc',val='undOvr')),
                   element('sub',index),element('sup',upper),element('e',body))
def root(body):return element('rad',element('radPr',element('degHide',val='1')),element('deg'),element('e',body))


def make_flow(path,boxes,loop=False):
    width=1600;boxheight=140;gap=52;left=210;right=1390
    image=Image.new('RGB',(width,len(boxes)*(boxheight+gap)+45),'white')
    d=ImageDraw.Draw(image)
    fonts=('C:/Windows/Fonts/arial.ttf','/mnt/c/Windows/Fonts/arial.ttf',
           '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(next(name for name in fonts if Path(name).exists()),35)
    for i,text in enumerate(boxes):
        y=25+i*(boxheight+gap)
        d.rounded_rectangle((left,y,right,y+boxheight),radius=15,fill='#F3F5F7',outline='#303A43',width=3)
        bounds=d.multiline_textbbox((0,0),text,font=font,spacing=7,align='center')
        d.multiline_text(((width-(bounds[2]-bounds[0]))/2,y+(boxheight-(bounds[3]-bounds[1]))/2-5),text,font=font,fill='black',align='center',spacing=7)
        if i<len(boxes)-1:
            bottom=y+boxheight;end=bottom+gap-8
            d.line((800,bottom,800,end),fill='black',width=4)
            d.polygon([(788,end-12),(812,end-12),(800,end+2)],fill='black')
    if loop:
        y1=25+(boxheight+gap)*1+boxheight/2;y2=25+(boxheight+gap)*3+boxheight/2
        d.line([(right,y2),(1515,y2),(1515,y1),(right+5,y1)],fill='#425D70',width=4)
        d.polygon([(right+15,y1-10),(right+15,y1+10),(right,y1)],fill='#425D70')
    image.save(path)


def main():
    if TARGET.exists():
        raise SystemExit('The current report has author formatting and editorial changes. Edit the DOCX in place; this former full-report generator will not overwrite it.')
    ASSETS.mkdir(exist_ok=True)
    make_flow(ASSETS/'main_flow.png',[
        'Original right wrist XYZ trajectory\nSame clip and fixed analysis windows',
        'Estimate detection speed on the full clip\nRetain prominent peaks and intervening minima',
        'Propose movement intervals and count K\nKeep nominal detector settings fixed',
        'Fit onset and completion near each interval\nSolve endpoint constrained linear LS for displacement',
        'Reconstruct and compare position and velocity\nUniform baselines at the same prescribed K',
        'Examine count and timing bound sensitivity\nReport errors and failure cases'])
    make_flow(ASSETS/'fitting_flow.png',[
        'Choose one of four initial timing vectors\nWithin the prescribed local timing bounds',
        'Evaluate a trial onset and completion vector\nConstruct the minimum jerk basis matrix',
        'Eliminate the final displacement\nSolve the remaining displacement vectors by linear LS',
        'Return raw XYZ position residuals to TRF\nUpdate timing and repeat until termination',
        'Retain every run and its solver status\nChoose the lowest RMSE eligible result'],loop=True)
    m=json.loads((ROOT/'results/mainline/summary.json').read_text())
    s=json.loads((ROOT/'results/timing_bound_sensitivity/summary.json').read_text())
    audit=json.loads((ROOT/'results/timing_bound_sensitivity/audit.json').read_text())
    control=json.loads((ROOT/'results/final_pipeline/summary.json').read_text())
    synthetic=json.loads((ROOT/'archive/third_stage/results/movement_unit_validation/summary.json').read_text())
    goals=list(csv.DictReader((ROOT/'results/mainline/goals.csv').open(newline='')))
    doc=Document();sec=doc.sections[0]
    sec.page_width=Inches(8.5);sec.page_height=Inches(11)
    sec.top_margin=Inches(.65);sec.bottom_margin=Inches(.65)
    sec.left_margin=Inches(.72);sec.right_margin=Inches(.72)
    for name in ('Normal','Title','Subtitle','Heading 1','Heading 2','Caption'):
        style=doc.styles[name];style.font.name='Times New Roman';style.font.color.rgb=RGBColor(0,0,0)
        style.font.underline=False;style.font.size=Pt(11)
        rpr=style.element.get_or_add_rPr();fonts=rpr.find(qn('w:rFonts'))
        if fonts is not None:
            for key in ('asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme'):fonts.attrib.pop(qn('w:'+key),None)
            fonts.set(qn('w:ascii'),'Times New Roman');fonts.set(qn('w:hAnsi'),'Times New Roman')
    doc.styles['Normal'].paragraph_format.space_after=Pt(6)
    doc.styles['Normal'].paragraph_format.line_spacing=1.06
    doc.styles['Title'].font.size=Pt(20)
    doc.styles['Subtitle'].font.size=Pt(12)
    doc.styles['Heading 1'].font.size=Pt(14);doc.styles['Heading 1'].font.bold=True
    doc.styles['Heading 2'].font.size=Pt(12);doc.styles['Heading 2'].font.bold=True
    for name in ('Heading 1','Heading 2'):
        doc.styles[name].paragraph_format.space_before=Pt(9)
        doc.styles[name].paragraph_format.space_after=Pt(6)
    doc.styles['Caption'].font.size=Pt(9.5)
    header=sec.header.paragraphs[0];header.text='WRIST MOTION DECOMPOSITION'
    header.runs[0].font.size=Pt(9)
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    doc.core_properties.author='Pin-Kuang Shih'
    doc.core_properties.title='Motion Decomposition with Fitted Minimum Jerk Goals'
    doc.core_properties.subject='Comprehensive third-stage experimental report and parameter sensitivity'
    figures=[];equations=[];tables=[]

    def p(text):return doc.add_paragraph(text)
    def heading(text,level=1):return doc.add_heading(text,level=level)
    def page(title):doc.add_page_break();heading(title)
    def equation(*items):
        number=len(equations)+1;equations.append(number)
        para=doc.add_paragraph();para.paragraph_format.space_before=Pt(5);para.paragraph_format.space_after=Pt(7)
        math=element('oMath',*items);para._p.append(math)
        para.add_run(f'    ({number})').font.size=Pt(10)
        return para
    def table(title,headers,rows,widths=None):
        number=len(tables)+1;tables.append(title)
        cap=doc.add_paragraph(f'Table {number}. {title}',style='Caption');cap.paragraph_format.keep_with_next=True
        tt=doc.add_table(rows=1,cols=len(headers));tt.alignment=WD_TABLE_ALIGNMENT.CENTER;tt.autofit=False
        widths=widths or [7.06/len(headers)]*len(headers)
        for col,w in zip(tt.columns,widths):col.width=Inches(w)
        for cell,value in zip(tt.rows[0].cells,headers):cell.text=str(value)
        for row in rows:
            for cell,value in zip(tt.add_row().cells,row):cell.text=str(value)
        for i,row in enumerate(tt.rows):
            prop=row._tr.get_or_add_trPr();prop.append(OxmlElement('w:cantSplit'))
            if i==0:prop.append(OxmlElement('w:tblHeader'))
            for j,cell in enumerate(row.cells):
                cell.width=Inches(widths[j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                cp=cell._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
                for side in ('top','left','bottom','right'):
                    edge=OxmlElement('w:'+side);edge.set(qn('w:val'),'single');edge.set(qn('w:sz'),'4');edge.set(qn('w:color'),'D9D9D9');borders.append(edge)
                cp.append(borders);margin=OxmlElement('w:tcMar')
                for side in ('top','bottom','left','right'):
                    e=OxmlElement('w:'+side);e.set(qn('w:w'),'65');e.set(qn('w:type'),'dxa');margin.append(e)
                cp.append(margin)
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E7EBEF' if i==0 else ('F8F9FA' if i%2==0 else 'FFFFFF'));cp.append(shade)
                for para in cell.paragraphs:
                    para.paragraph_format.space_after=Pt(2);para.paragraph_format.line_spacing=1.0
                    if j>0:para.alignment=WD_ALIGN_PARAGRAPH.CENTER
                    for rr in para.runs:rr.font.size=Pt(9.5);rr.bold=i==0
        doc.add_paragraph().paragraph_format.space_after=Pt(1)
    def figure(path,caption,width=6.8):
        path=Path(path)
        if path.parent!=ASSETS and (ASSETS/path.name).exists():path=ASSETS/path.name
        assert path.exists();assert str(path) not in [x['path'] for x in figures]
        para=doc.add_paragraph();para.alignment=WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next=True;para.paragraph_format.space_after=Pt(3)
        image=para.add_run().add_picture(str(path),width=Inches(width))
        image._inline.docPr.set('descr',caption)
        number=len(figures)+1;figures.append({'number':number,'path':str(path),'caption':caption,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        cap=doc.add_paragraph(f'Figure {number}. {caption}',style='Caption');cap.paragraph_format.space_after=Pt(7)
    def code(lines):
        for line in lines:
            para=doc.add_paragraph();para.paragraph_format.space_after=Pt(1)
            rr=para.add_run(line);rr.font.name='Consolas';rr.font.size=Pt(9)

    doc.add_paragraph('Motion Decomposition with Fitted Minimum Jerk Goals',style='Title')
    doc.add_paragraph('Speed based count proposals and endpoint constrained trajectory fitting',style='Subtitle')
    p('Pin-Kuang Shih | Third-stage experimental report | Editable draft | 5 November 2026')
    heading('Abstract')
    p('I investigated how to convert a recorded wrist trajectory into overlapping minimum-jerk displacement components. Unlike the earlier sampled-goal approach, I fit displacement vectors by linear least squares for each trial timing configuration and optimize component onset and completion times. Prominent speed lobes propose the number of components, while intervening speed minima define candidate movement intervals. Local timing neighborhoods preserve an association between components and these intervals while allowing overlap. These neighborhoods are heuristic constraints, not physiological limits.')
    p('The final main experiment analyzes two three-second windows from the same AMASS ACCAD conversation-gesture clip. The nominal detector proposes four and three components. Position RMSE is 4.194 cm and 3.022 cm, respectively. Both are lower than the same-count uniform baselines. Velocity RMSE is worse than both baselines in the first window and better in the second. Count and timing-bound sensitivity reveal substantial dependence on the proposed intervals and constraints. The report therefore presents a completed small-scale experiment, not a unique or generally validated solution for identifying movement units.')
    p('I also document the earlier free-timing and penalty-based explorations, including large cancellation in a free fit and a duplicate-splitting defect in the custom opposition penalty. Those results explain design decisions but are not pooled with the final main method. All numerical tables below are generated from saved experimental outputs; native Word equations and text remain editable.')
    table('Reading guide',['Sections','Purpose'],[
        ('1–2','Problem, scope and prior explorations'),('3–7','Model, least squares, detection, timing and solver'),
        ('8–14','Data, metrics, main results, count and timing sensitivity'),('15–18','Verification, limitations, conclusions and reproduction'),('Appendices A–C','Numerical detail, goal sequences and references')],[1.1,5.96])

    page('1 Research objective and scope')
    p('The practical objective is to describe one observed wrist trajectory using a sequence of overlapping minimum-jerk goals. The central unknowns are the component count, onset times, completion times and displacement vectors. A good position fit alone is insufficient: redundant or opposing components may reproduce the trajectory without a useful interpretation.')
    p('The motivating feedback was to fit goal positions rather than copying samples, derive timing and count from motion, compare position and velocity errors against a fixed-interval baseline across counts, and explain remaining failures. Speed minima were suggested as a more natural movement-boundary cue than acceleration peaks. This report addresses those points with a reproducible heuristic proposal and fitting procedure; it does not establish true neural movement units.')
    table('Requested elements and the evidence provided',['Element','Implementation or evidence'],[
        ('Fitted goal positions','Endpoint-constrained linear least squares inside every timing evaluation'),
        ('Motion related timing','Speed-minimum interval proposals and bounded continuous timing optimization'),
        ('Motion related count','One component per retained speed lobe; count sensitivity is reported'),
        ('Errors across counts','All unique detected partitions in a prespecified prominence sweep'),
        ('Uniform comparison','Sampled-goal and fitted-displacement controls at the same K'),
        ('Explanation and failures','Velocity deterioration, boundary sensitivity and prior degeneracy are retained')],[1.75,5.31])
    p('The final scope is deliberately limited to 18–21 s and 30–33 s from the same source clip. Each crop is reconstructed independently with its observed initial and final positions. This is not a new reconstruction of the entire clip, a multi-subject benchmark, a full-body controller or a learned goal generator. No physics engine is required because the experiment evaluates kinematic trajectory reconstruction, not dynamic feasibility.')
    heading('Terminology',2)
    p('K denotes the number of additive displacement components and, by cumulative addition, K goals. The initial position is separate and is not counted as a goal. A component is a model element, not necessarily a complete semantic action. Multiple straight components may be needed to describe one curved movement [1].')

    page('2 Model background and development of the experiment')
    p('The overlapping representation follows the motion-generation setting of Sato et al. [1]. The classical minimum-jerk model supplies a smooth point-to-point movement profile [2]. These references motivate the representation; they do not make an arbitrary fitted decomposition physiologically unique. Liao and Kirsch [3] provide a related example of fitting additive minimum-jerk submovements and considering opposing contributions. My earlier custom penalty and count rule were not full reproductions of that work.')
    table('Explorations and their role in this report',['Stage','What was tried','What was learned'],[
        ('Initial experiment','Uniformly sampled goals and overlapping displacements','Useful implementation baseline; count and timing unrelated to the motion'),
        ('First follow-up','Error-driven goal insertion and acceleration-based sampling','Did not fit the core parameters or establish movement-unit structure'),
        ('Third-stage exploration','Fitted positions and times; custom opposition penalty; cost-curve knee','Improved fits but introduced degeneracy, count-rule and penalty-design concerns'),
        ('Final main experiment','Speed-lobe proposals, local time association and linear LS','Produces explicit motion-related proposals; remaining fit and sensitivity failures are measurable')],[1.15,2.9,3.01])
    p('Acceleration peaks were not retained as movement boundaries. For a smooth point-to-point component, acceleration changes occur within the movement, so a large acceleration value need not mark a separation between movements. The final detector instead looks for speed structure; even so, a shallow minimum can still lie inside one movement and overlap can hide a boundary.')
    p('I separated the final main method from historical controls rather than mixing whichever result had the smallest error. In particular, the old penalty-based knee candidates were K=3 and K=4; the current speed-based nominal proposals are K=4 and K=3. These are different procedures and must not be treated as interchangeable estimates of a true count.')
    heading('What is new in the final main method',2)
    p('Positions are fitted rather than sampled. Both onset and completion are optimized rather than fixed at uniform intervals. The count proposal depends on recorded speed structure. Local timing association replaces the earlier custom cancellation penalty in the main method. An explicit three-setting sensitivity experiment tests this new restriction without changing the detected counts.')

    page('3 Overlapping minimum jerk representation')
    table('Notation',['Symbol','Meaning','Unit'],[
        ('tₙ, N','Sample time and number of samples','s, count'),('pₙ, p̂(t)','Observed and reconstructed XYZ position','m'),
        ('p₀, pF','Fixed observed crop endpoints','m'),('K','Number of displacement components and cumulative goals','count'),
        ('aᵢ, bᵢ','Onset and completion of component i','s'),('Tᵢ = bᵢ − aᵢ','Component duration','s'),
        ('dᵢ','Fitted three-dimensional displacement vector','m'),('Sᵢ(t), uᵢ(t)','Progress and clipped normalized time','dimensionless'),
        ('gᵢ','Cumulative goal after component i','m')],[1.05,4.8,1.21])
    p('For one isolated point-to-point reach, jerk is the third time derivative of position. The classical minimum-jerk problem minimizes its integrated squared magnitude, with prescribed endpoint positions and zero endpoint velocity and acceleration [2]. The quintic profile below is the resulting basis used here; fitting several such bases is a separate problem.')
    equation(sub('J','jerk'),' = ',element('nary',element('naryPr',element('chr',val='∫'),element('limLoc',val='subSup')),element('sub','a'),element('sup','b'),element('e',sup(seq('‖',frac(seq(sup('d','3'),'p'),seq('d',sup('t','3'))),'‖'),'2'),' dt')))
    equation(sub('T','i'),' = ',sub('b','i'),' − ',sub('a','i'),' > 0')
    equation(sub('u','i'),'(t) = min(1, max(0, ',frac(seq('t − ',sub('a','i')),sub('T','i')), '))')
    equation(sub('S','i'),'(t) = 10',sup(sub('u','i'),'3'),' − 15',sup(sub('u','i'),'4'),' + 6',sup(sub('u','i'),'5'))
    equation(hat('p'),'(t) = ',sub('p','0'),' + ',total('i = 1','K',seq(sub('S','i'),'(t)',sub('d','i'))))
    p('Before onset, progress is zero; after completion, progress is one. The quintic gives zero component velocity and acceleration at both ends. Components add as vectors and may be active simultaneously. Their sum can form a curved path even though each individual displacement has a fixed direction. The experiment fits this representation; it does not minimize the jerk of the total observed motion as a separate global objective.')
    equation(sub('v','i'),'(t) = ',frac(seq('30',sup(sub('u','i'),'2'),sup(seq('(1 − ',sub('u','i'),')'),'2')),sub('T','i')),sub('d','i'))
    equation(sub('g','i'),' = ',sub('p','0'),' + ',total('j = 1','i',sub('d','j')))
    p('The velocity expression is zero outside the active interval because normalized time is clipped. A cumulative goal is a representation parameter: while components overlap, the reconstructed wrist need not pass through every intermediate cumulative goal. It should not be interpreted as an observed contact point or a full-body pose.')

    page('4 Fitting goal positions by linear least squares')
    p('When the timings are fixed, each progress value is known. Define an N by K matrix B, whose n,i entry is the progress of component i at sample n. Stack displacements into a K by 3 matrix D. Each Cartesian coordinate then shares the same basis and is fitted simultaneously through a linear least-squares solve.')
    equation(sub('B','ni'),' = ',sub('S','i'),'(',sub('t','n'),')')
    equation('minimize over D    ',sup('‖BD − Y‖','2'),' ,     ',sub('Y','n'),' = ',sub('p','n'),' − ',sub('p','0'))
    p('Here the matrix norm means the sum of squared entries (the squared Frobenius norm). In the code, numpy.linalg.lstsq is used rather than forming an explicit inverse. The observed endpoints are enforced exactly by eliminating the final displacement. Because every component is inactive at the first sample and complete at the last sample, the displacements must sum to the endpoint difference.')
    equation('Δ = ',sub('p','F'),' − ',sub('p','0'),' ,     ',sub('d','K'),' = Δ − ',total('i = 1','K − 1',sub('d','i')))
    equation(sub('F','ni'),' = ',sub('B','ni'),' − ',sub('B','nK'),' ,     ',sub('y','n'),' = ',sub('p','n'),' − ',sub('p','0'),' − ',sub('B','nK'),'Δ')
    equation('minimize over ',sub('D','free'),'    ',sup(seq('‖F',sub('D','free'),' − y‖'),'2'))
    p('After solving the K−1 free displacement vectors, the last vector is recovered from the sum constraint. For K=1 there is no free displacement: d₁ is simply the endpoint difference, while onset and completion remain unknown. This directly answers why fitted goals can differ from sampled positions: they are chosen to explain all recorded samples jointly.')
    table('A small known-timing example',['Quantity','First component','Second component'],[
        ('Onset and completion','0.20 to 2.20 s','1.20 to 3.20 s'),('Known displacement XYZ m','(0.30, 0.10, −0.05)','(−0.12, 0.20, 0.08)'),
        ('At t=1.20 s','Progress 0.5','Progress 0'),('At t=2.20 s','Progress 1','Progress 0.5')],[2.05,2.5,2.51])
    p('For example, the two X-coordinate equations are 0.5 d₁x = 0.15 and d₁x + 0.5 d₂x = 0.24, giving 0.30 and −0.12 m. The actual implementation uses all samples, not just these two equations. Rank and conditioning are saved because distinct parameterizations can still generate indistinguishable trajectories.')

    page('5 Motion related count and interval proposals')
    p('The detector is an offline proposal mechanism. I first estimate XYZ velocity from the entire recorded clip with a third-order Savitzky–Golay derivative filter [6]. I take the vector norm and only then crop the analysis window. This avoids introducing a new filtering boundary at each crop. The fitting target remains the unfiltered recorded positions.')
    equation(sub('s','n'),' = ‖',sub('v','SG'),'(',sub('t','n'),')‖')
    p('The nominal window contains 17 samples at 120 Hz; the interval from its first to last sample is 16/120 = 0.133 s. Candidate speed peaks are retained when their prominence is at least 10% of the maximum detection speed in the crop, with peak separation at least 0.15 s. Prominence measures how far a peak stands above its surrounding baseline; it is not a requirement that speed exceed a fixed absolute value.')
    p('Between every pair of consecutive retained peaks, the lowest speed sample becomes a candidate boundary; an exact tie uses the earliest sample. The crop endpoints complete the partition. If P peaks are retained, K=max(1,P). No detected peak falls back to one component rather than proving that the crop contains a single movement.')
    equation('K = max(1, P)')
    table('Nominal detector settings',['Parameter','Value','Role'],[
        ('Derivative filter','17 samples, polynomial order 3','Suppress small fluctuations in the detection signal'),
        ('Relative prominence','0.10 × crop maximum detection speed','Retain distinct speed lobes'),
        ('Minimum peak separation','0.15 s','Avoid retaining very nearby peaks'),
        ('Boundary between peaks','Minimum speed sample','Propose a transition between candidate lobes'),
        ('Count fallback','K=1 when no peak exists','Always define a minimal proposal')],[1.9,2.0,3.16])
    p('The nominal partitions are 18.0000, 19.7167, 19.9500, 20.3083 and 21.0000 s (K=4), and 30.0000, 31.5333, 32.1750 and 33.0000 s (K=3). These settings are explicit heuristics, not standard physiological thresholds. One retained speed lobe is an initial unit hypothesis; it need not equal one actual movement unit.')
    p('The same three-second crop can yield a different count after changing the detection settings. Overlap may merge lobes, a shallow valley can split a single movement, and coordinate jitter can add speed fluctuations. These are reasons to report sensitivity rather than treating the nominal count as ground truth.')

    page('6 Timing neighborhoods and their interpretation')
    p('A fully free timing fit can place several components in nearly identical time ranges and produce large opposing displacements. To retain an association with the proposed speed structure, the main method assigns one component to each candidate interval and constrains its onset and completion near the interval boundaries. It still permits continuous timing adjustment and overlap.')
    p('Let the interval boundaries be q₀ through qK and let ℓᵢ=qᵢ−qᵢ₋₁. At an internal boundary qⱼ, use rⱼ=min(ℓⱼ,ℓⱼ₊₁); at either crop edge use the adjacent interval width. The local scales at the two ends of a component can differ. With outward fraction α and inward fraction β, the allowed times are:')
    equation('max(',sub('t','0'),', ',sub('q','i−1'),' − α',sub('r','i−1'),') ≤ ',sub('a','i'),' ≤ ',sub('q','i−1'),' + β',sub('r','i−1'))
    equation(sub('q','i'),' − β',sub('r','i'),' ≤ ',sub('b','i'),' ≤ min(',sub('t','F'),', ',sub('q','i'),' + α',sub('r','i'),')')
    p('The nominal α=0.45 controls outward extension: an onset can move earlier and a completion can move later. The nominal β=0.20 controls inward contraction: an onset can move later and a completion can move earlier. Fractions are dimensionless, not seconds or error tolerances. For β<0.5, the local bounds keep onset before completion because each boundary scale is no larger than the component interval width.')
    table('Worked example without crop clipping',['Quantity','Allowed interval'],[
        ('Candidate interval','L=2 s, R=3 s; rL=rR=0.5 s'),('Onset','2 − 0.45×0.5 to 2 + 0.20×0.5 = 1.775 to 2.100 s'),
        ('Completion','3 − 0.20×0.5 to 3 + 0.45×0.5 = 2.900 to 3.225 s')],[1.45,5.61])
    p('The first initialization extends each boundary outward by 0.25 times its local scale, clipped to the crop. The 0.25 initial extension is distinct from the 0.45/0.20 feasible bounds. Three further initializations use seed 23 and normalized random draws between 0.05 and 0.95 of each allowable interval.')
    p('The reason for local association is interpretable component placement, not a proven biological duration limit. The numerical coefficients have no unique theoretical justification. Their influence is tested explicitly in Section 14; many fitted times touch the bounds, so these constraints must not be described as innocuous.')

    page('7 Complete algorithm and nested optimization')
    figure(ASSETS/'main_flow.png','The final main pipeline. Count proposals are derived from detection speed; displacement and timing are fitted. Evaluation and sensitivity do not replace the nominal proposal by the lowest-error setting.',width=5.6)
    p('The detector is run once for a prescribed setting. The fitting stage neither inserts goals at large-error samples nor retunes the count from the reconstruction error. This preserves a distinction between a motion-based proposal and the quality of the subsequent approximation.')
    p('All fitted outputs, termination statuses and candidate partitions are retained. A successful numerical termination only qualifies a result for comparison; it does not certify a global optimum, true movement units or physical feasibility.')

    page('7 Continued Time optimization with linear displacement fitting')
    figure(ASSETS/'fitting_flow.png','The inner fitting procedure. TRF updates only timing; every residual evaluation recomputes the best endpoint-constrained linear displacement fit. The side arrow denotes repeated timing evaluations.',width=5.1)
    equation('minimize over a,b    ',total('n = 1','N',sup(seq('‖',hat('p'),'(',sub('t','n'),'; a,b, ',sup('D','*'),'(a,b)) − ',sub('p','n'),'‖'),'2')))
    p('SciPy least_squares uses the trust-region reflective method, a three-point finite-difference Jacobian, x_scale="jac", linear loss, ftol=xtol=gtol=10⁻¹² and max_nfev=1000 [5]. The returned XYZ residuals are unweighted; minimizing their squared sum is equivalent to minimizing position MSE. No velocity objective or opposition penalty is included in the final method.')
    p('Four starts are fitted per condition. The selected result is the lowest position RMSE among successful, finite, endpoint-valid results, with the lower start index breaking a tie. A failed or budget-limited run is retained rather than silently treated as converged. Linear displacement fitting is exact for each fixed timing trial, but the outer timing optimization remains local.')

    page('8 Data evaluation and comparison design')
    p('The source is Female1Gestures_c3d/D3 - Conversation Gestures_poses.npz from the ACCAD subset of AMASS [4]. The existing extraction uses SMPL+H to obtain the right-wrist joint in global XYZ coordinates, including the stored global translation. The source motion has 6086 frames at 120 Hz. Both analysis windows include their endpoints and contain 361 samples. The intermediate wrist file is data/wrist_trajectory.npz.')
    p('Initial and final positions are fixed to the observed crop endpoints. This positional constraint should not be confused with measured zero endpoint velocity: the additive model forces zero analytical component velocity at the crop edges, while the measured wrist may still be moving. Global wrist motion also includes body motion; this report does not isolate a hand-relative or torso-relative reaching task.')
    equation(sub('RMSE','p'),' = ',root(frac(total('n = 1','N',sup(seq('‖',hat('p'),'(',sub('t','n'),') − ',sub('p','n'),'‖'),'2')),'N')))
    equation(sub('RMSE','v'),' = ',root(frac(total('n = 1','N',sup(seq('‖',hat('v'),'(',sub('t','n'),') − ',sub('v','n'),'‖'),'2')),'N')))
    equation(sub('E','max'),' = ',sub('max','n'),' ‖',hat('p'),'(',sub('t','n'),') − ',sub('p','n'),'‖')
    p('For evaluation, both original and reconstructed velocity are computed with numpy.gradient using the same timestamps and edge_order=2. Thus the velocity metric compares like-for-like numerical derivatives, rather than comparing a filtered reference with an analytical fitted velocity. The three-dimensional distance is evaluated per time sample; RMSE is not an average of three independent axis RMSE values. Stored units are meters and meters per second; position tables convert meters to centimeters.')
    table('Comparison conditions',['Method','Timing','Displacement'],[
        ('Uniform sampled','K equal intervals with 0.25 overlap extension','Goals sampled at uniform interval ends'),
        ('Uniform fitted','Same fixed times as uniform sampled','Endpoint-constrained linear least squares'),
        ('Valley fitted','Locally optimized around detected intervals','Linear least squares at each trial time')],[1.35,2.9,2.81])
    p('The comparisons use the same raw target, K, model and positional endpoints. The adaptive method uses four nonlinear starts, while the fixed-time fitted control needs one linear solve; total computation is not equal. Different search spaces and initialization budgets prevent attributing every difference solely to the use of valleys. No method is claimed to reach a global optimum.')

    page('9 Nominal count proposals and main results')
    rows=[]
    names={'uniform_sampled':'Uniform sampled','uniform_fitted':'Uniform fitted','valley_fitted':'Valley fitted'}
    for label in ('18_21','30_33'):
        k=m['details'][label]['detection']['component_count']
        for method in ('uniform_sampled','uniform_fitted','valley_fitted'):
            r=next(r for r in m['rows'] if r['window']==label and r['K']==k and r['method']==method)
            rows.append((label.replace('_','–'),names[method],k,f"{100*r['position_rmse_m']:.3f}",f"{r['velocity_rmse_m_s']:.4f}",f"{100*r['max_position_error_m']:.3f}"))
    table('Same count results at nominal detector settings',['Window s','Method','K','Pos RMSE cm','Vel RMSE m/s','Max pos cm'],rows,[.8,1.35,.35,1.45,1.5,1.61])
    figure(ROOT/'results/final_pipeline/count_selection.png','Nominal detection speed, retained peaks and candidate speed-minimum boundaries. The first window proposes K=4 and the second K=3. These are heuristic proposals, not known labels.',width=6.6)
    p('Position RMSE improves relative to both same-count baselines in both windows. The first window, however, has higher velocity RMSE than either baseline and higher maximum position error than the uniform fitted control. The second window improves all three reported metrics. Therefore the result is not a uniform improvement across windows or metrics.')

    page('10 First window reconstruction and failure case')
    figure(ROOT/'results/mainline/trajectories_18_21.png','First window at K=4. Original and reconstructed XYZ position are shown on the left; numerical XYZ velocity is shown on the right. All methods share the raw target and positional endpoints.',width=6.85)
    p('The nominal valley method lowers position RMSE from 5.254 cm for the fitted uniform control to 4.194 cm, but raises velocity RMSE from 0.3160 to 0.3620 m/s. Its sharp fitted velocity excursion does not match the broad timing of the reference movement. The maximum position error rises from 13.265 to 14.420 cm even though mean squared position error falls.')
    p('The 19.7167 s boundary is a relative minimum within a high-speed region, rather than a clear stop. Treating the neighboring lobes as separate units with local timing constraints can restrict the reconstruction. The position-only objective also does not guarantee a better velocity fit. These are plausible explanations supported by the plots; the experiment does not separately quantify their causal contributions.')

    page('11 Second window reconstruction and remaining error')
    figure(ROOT/'results/mainline/trajectories_30_33.png','Second window at K=3 using the same evaluation convention. The approximation follows the major Y and Z movement more closely than either same-count uniform control, but misses some X and earlier trajectory details.',width=6.85)
    p('Position RMSE falls from 7.535 cm for the fitted uniform control to 3.022 cm, velocity RMSE from 0.4201 to 0.2631 m/s, and maximum position error from 17.491 to 8.239 cm. This is a clear improvement for this crop under the tested conditions, not proof that the proposed three components are the correct movement units.')
    p('The nominal three-component approximation still misses the X-position excursion around 31.8 s and some earlier shape changes. The count sweep finds lower position error with additional detected partitions. I retain the nominal count rather than increasing it after observing this failure, because count sensitivity and nominal performance answer different questions.')

    page('12 First window component timing and interpretation')
    figure(ROOT/'results/mainline/components_18_21.png','First-window detection speed and individual analytical component speed magnitudes. Magnitudes are shown for interpretation; they must not be added as scalars to obtain the total speed.',width=6.55)
    table('First window fitted timing and displacement',['i','Onset s','End s','Duration s','Displacement XYZ m'],[
        (i+1,f'{a:.4f}',f'{b:.4f}',f'{b-a:.4f}',' / '.join(f'{v:.4f}' for v in d))
        for i,(a,b,d) in enumerate(zip(m['details']['18_21']['fit']['activation'],m['details']['18_21']['fit']['completion'],m['details']['18_21']['fit']['displacements']))],[.35,1.05,1.05,1.1,3.51])
    p('The component displacement-length sum is 0.594 m. This quantity is a diagnostic of the internal representation, not the physical distance traveled by the wrist; vector overlap and cancellation change the path of the sum. The narrow second component and its high speed are consistent with the observed velocity failure. Six of the eight timing parameters lie close to a feasible bound in the nominal fit, including possible crop-edge constraints.')

    page('13 Second window component timing and free fit control')
    figure(ROOT/'results/mainline/components_30_33.png','Second-window detection speed and component speed magnitudes. Overlap is allowed by the fitted onset and completion times; no requirement is imposed that the wrist stop at every proposed boundary.',width=6.55)
    table('Second window fitted timing and displacement',['i','Onset s','End s','Duration s','Displacement XYZ m'],[
        (i+1,f'{a:.4f}',f'{b:.4f}',f'{b-a:.4f}',' / '.join(f'{v:.4f}' for v in d))
        for i,(a,b,d) in enumerate(zip(m['details']['30_33']['fit']['activation'],m['details']['30_33']['fit']['completion'],m['details']['30_33']['fit']['displacements']))],[.35,1.05,1.05,1.1,3.51])
    p('The final local-association method has a displacement-length sum of 0.501 m. In the separate free-timing, unpenalized K=3 control, the selected finite fit has position RMSE 2.339 cm but a displacement-length sum of 212.456 m. Its lower positional error does not justify an anatomical interpretation. Local timing association avoids that extreme behavior in this example; it is not a general guarantee against opposing components.')
    p('The free-timing control uses four starts for each prescribed K from 1 to 12 in both windows: 96 run records, of which four failed numerical termination and were excluded from eligible selection. Each K still has an eligible result. The control is saved separately and is not substituted for the main method when it achieves a lower error.')

    page('14 Count and timing constraint sensitivity')
    figure(ROOT/'results/mainline/errors_vs_count.png','Errors for every distinct detected partition in the count sweep, with uniform baselines at the same K. Black stars mark the nominal detector setting. The detected partition changes along with K, so the points are not a pure count-only experiment.',width=6.65)
    p('For count sensitivity I keep the 17-sample filter and 0.15 s peak gap, and sweep relative prominence from 0.01 to 0.40 in increments of 0.01. Identical boundary partitions are fitted once. This yields eight partitions in the first window and six in the second, with four fits per partition: 56 successful runs in total. The corresponding same-count comparisons contain 30 rows. Appendix A lists every partition result.')
    p('The first window improves substantially when the detector proposes at least five or six components. The second improves from three to four components but changes little in position error at six and eight. This describes the tradeoff under the specified proposals; it does not identify a universally correct count. Because the local feasible timing regions change with the partition, error need not be monotonic in K.')

    page('14 Continued Timing neighborhood sensitivity')
    figure(ROOT/'results/timing_bound_sensitivity/sensitivity.png','Timing-bound sensitivity at fixed nominal counts and boundaries. Narrow, nominal and wide mean (α,β)=(0.30,0.10), (0.45,0.20) and (0.60,0.30). The vertical scales differ between windows.',width=6.5)
    table('Sensitivity results with fixed count and boundaries',['Window s','α / β','Pos cm','Vel m/s','Max cm','Bound hits'],[
        (r['window'].replace('_','–'),f"{r['outward']:.2f} / {r['inward']:.2f}",f"{100*r['position_rmse_m']:.3f}",f"{r['velocity_rmse_m_s']:.4f}",f"{100*r['max_position_error_m']:.3f}",f"{r['bound_hits']} / {2*r['K']}") for r in s['rows']],[.85,1.15,1.2,1.25,1.2,1.41])
    p('All 24 fits terminate successfully and the two nominal results reproduce the main reconstruction to an absolute 10⁻¹⁰ m check tolerance. A bound hit means a time is within 10⁻⁶ of its feasible width from an edge; hits can also involve a crop endpoint. The experiment varies outward and inward fractions together, so their individual effects are not isolated.')

    page('14 Continued What the sensitivity supports')
    p('The nested neighborhoods were specified before the new fitting run. The first initialization has the same 0.25-scale outward extension in all settings; the other three use identical normalized random draws mapped into each box. Counts, proposed boundaries, data, objective and per-fit budget remain fixed. This controls the movement proposal while testing how strongly the time restrictions affect its reconstruction.')
    p('The wider neighborhood lowers first-window position RMSE by about 19% relative to nominal and second-window position RMSE by about 5%. Many timing parameters are at or near bounds. Thus I cannot claim that the numerical coefficients are unimportant. A larger feasible region permits more timing flexibility; finite multistart fitting still does not guarantee a global optimum or monotonically improving results in general.')
    p('Across the three tested settings, the first window has lower position RMSE but higher velocity RMSE than both uniform baselines. The second has lower position and velocity RMSE than both. These main directions survive this limited sensitivity check. Maximum position error is less consistent: the wider first-window fit improves it relative to the fixed-time fitted baseline, while the nominal fit does not.')
    p('I retain the original 0.45/0.20 condition in the main tables rather than selecting the wider condition after observing lower error. There is no independent validation set, external accuracy requirement or justified coefficient-selection rule that establishes an optimal neighborhood. The experiment covers only these two crops and the nominal detected counts, not every count partition or subject.')
    table('Detector count sensitivity retained from the main evidence',['Filter samples','First window K at 0.05 / 0.10 / 0.20','Second window K at 0.05 / 0.10 / 0.20'],[
        (w,' / '.join(str(next(r['K'] for r in m['detector_sensitivity'] if r['window']=='18_21' and r['filter_samples']==w and r['prominence']==pr)) for pr in (.05,.1,.2)),
         ' / '.join(str(next(r['K'] for r in m['detector_sensitivity'] if r['window']=='30_33' and r['filter_samples']==w and r['prominence']==pr)) for pr in (.05,.1,.2))) for w in (9,17,25)],[1.1,2.98,2.98])
    p('This grid measures detection counts only; it is not a grid of fitted accuracies. It shows why the nominal K=4/3 proposal cannot be presented as stable ground truth. The more extensive 0.01–0.40 count experiment fits every unique nominal-filter partition and reports its results separately.')

    page('15 Numerical verification and known answer experiments')
    p('I distinguish a correct implementation from a correct interpretation. The final code has 54 passing numerical tests, including a one-component detection and recovery case, a known overlapping two-component fit, endpoint preservation, invalid partitions, and agreement between the nominal sensitivity fitter and the existing main fitter. These are controlled checks, not a success-rate estimate on human movement.')
    p('The independent report audit reconstructs trajectories using a separately written quintic polynomial, recomputes position and velocity metrics, checks bounds and eligible-result selection, compares CSV and NPZ data, checks the seven exported goals, and verifies original data and source-code hashes. It covers 30 main comparison rows, 56 local-time fits and 24 sensitivity fits. Source consistency does not make the physiological hypothesis true.')
    table('Known answer checks and what they establish',['Check','Result','Scope'],[
        ('Single clean component','K=1 can be proposed and fitted','Avoids the old knee rule excluding K=1'),
        ('Known overlapping pair','Trajectory and displacement recovered within test tolerances','Checks fitter when the supplied interval association is compatible'),
        ('Fixed timing linear LS','Known generated displacements can be recovered','Checks the linear algebra, not count inference'),
        ('Nominal sensitivity condition','Main reconstruction reproduced within 1e−10 m','Checks comparison consistency'),
        ('Original XYZ source and goals','Data and exported sequences match saved evidence','Checks reproducibility and reporting')],[1.65,2.8,2.61])
    p('A separate, earlier 480-record structural validation concerned the old penalty-and-knee pipeline, not this final local-association method. At a supplied generating count, clean single and separated two-component cases were recovered to numerical precision; the penalized overlapping case had about 0.0127 mm position RMSE. Nevertheless, the old automatic count pipeline abstained for all four cases when the upper count was 12. Those findings motivate separating fitting accuracy from count inference and are not a validation of the final count rule.')
    p('The saved earlier synthetic cases include clean single, separated pair, overlapping pair and a clean component plus controlled oscillation. A perturbed observation is not declared to have exact count one. Coincident time bases can also be algebraically indistinguishable from a single summed displacement. The report does not demand that an unidentifiable generating count be recoverable from positions alone.')

    page('16 Historical count and penalty diagnostics')
    p('The earlier cost-curve knee rule normalized K and fitted cost and chose the greatest positive interior distance below the endpoint chord. It could not choose the lowest candidate K by construction, and a strict nonmonotonicity guard often returned no count. Real-window K=3/4 candidates from this rule are historical outputs, not the count proposals of the final method.')
    p('The custom opposition penalty used negative component-velocity dot products. With reference duration T and length L, qᵢⱼ=max(0,−vᵢ·vⱼ), and the penalty was a sum of squared scaled pairwise terms. This was my implementation inspired by the cancellation concern in [3], not a reproduction of that paper’s objective.')
    equation(sub('P','opp'),' = ',frac('1','N'),total('n = 1','N',total('i < j','',sup(seq('(',frac(sup('T','2'),'L'),sub('q','ij'),'(',sub('t','n'),'))'),'2'))))
    p('An exact proportional-splitting check exposed a design defect: replacing every component by m identical-time copies with displacement dᵢ/m leaves total position and velocity unchanged, yet reduces this squared pairwise penalty by a factor m². Same-direction copies add no new negative pair terms. Thus lower regularized cost across counts could reflect redundant representation rather than more meaningful structure.')
    table('Exact splitting diagnostic for one overlapping pair',['Copies per component','Total K','Opposition cost','λP with λ=1e−6'],[
        (1,2,'0.03966319','3.9663e−8'),(2,4,'0.00991580','9.9158e−9'),(4,8,'0.00247895','2.4789e−9')],[1.65,.75,2.3,2.36])
    p('The reconstructed curves agree within 10⁻¹⁴ m. This does not prove that a particular real fit exploited the loophole, nor does it invalidate its measured trajectory error. It does mean the penalty is unsuitable as unqualified structural evidence. The final main method therefore excludes it; this report does not claim a corrected universal opposition penalty.')
    p('The free, unpenalized fitting control also shows why merely removing a flawed penalty is insufficient. In the second nominal crop, very large opposing displacements can improve position fit. Local association is a transparent model restriction tested here, with a measurable cost in flexibility and no guarantee of biological correctness.')

    page('17 Limitations and response to the research objective')
    heading('Count proposals remain heuristic',2)
    p('Speed minima provide a motion-related cue, but the number of retained lobes depends on smoothing, prominence and separation. The count sweep gives a tradeoff rather than a principled final-count optimum. A curved movement may require several straight components and overlap can hide constituent movements. The method does not establish a one-to-one mapping between lobes, goals and human movement units.')
    heading('Timing association can distort the approximation',2)
    p('The local neighborhoods preserve an interpretable relation to candidate intervals, but they also prevent some good free fits. The first crop contains a high-speed shallow valley and produces a sharp velocity mismatch. The sensitivity experiment shows that changing the neighborhood materially changes the error. The chosen coefficients are empirical, not physiologically justified constants.')
    heading('The fitting objective does not include velocity',2)
    p('All main fits minimize position squared error. Velocity error is reported as a separate outcome, so its deterioration is a meaningful negative result rather than a numerical contradiction. A velocity-based or mixed objective would constitute a different method and is not silently added to this report.')
    heading('Model and data restrictions',2)
    p('The model describes global wrist kinematics, not contact, forces, muscle activation, joint limits, balance or whole-body feasibility. Independent crop reconstruction imposes zero analytical endpoint velocity despite nonzero measured speed. Raw numerical differentiation also exposes small trajectory fluctuations. Only two crops from one clip were examined, so no population-level or full-clip generalization claim is supported.')
    heading('What the experiment does establish',2)
    p('The implementation fits goal positions and timing, proposes count from motion rather than fixed intervals or error alone, and compares reconstruction errors with same-count uniform baselines. It records every tested partition, sensitivity result and failure. These are concrete responses to the requested experiment. They do not establish that count selection or meaningful movement decomposition is fully solved.')

    page('18 Conclusions and reproducibility')
    p('I completed a small-scale conversion experiment using overlapping minimum-jerk components with fitted timing and displacement. Endpoint-constrained linear least squares resolves the sampled-goal limitation. A speed-based proposal gives an explicit motion-related count and timing hypothesis, and local neighborhoods allow fitted overlap while maintaining that association.')
    p('The nominal K=4/3 results improve position RMSE in both windows against same-count uniform baselines, but velocity RMSE improves only in the second. Count and timing-bound sensitivity show that the proposals and constraints remain consequential design choices. I therefore report the approach as a tested heuristic with identifiable failure cases, rather than a finished general solution for extracting true movement units.')
    table('Evidence locations relative to the repository root',['Location','Contents'],[
        ('results/mainline/summary.json','All partitions and local-time run details'),('results/mainline/fits.npz','Original crop targets and reconstructed trajectories'),('results/mainline/metrics.csv','30 same-count main comparison records'),
        ('results/mainline/count_sweep.csv','14 distinct partition results'),('results/mainline/goals.csv','Seven fitted nominal components and cumulative goals'),
        ('results/timing_bound_sensitivity/','Protocol, 24 fits, metrics, figures and independent audit'),('results/final_pipeline/','Free-time control K=1–12 and uniform controls'),
        ('archive/third_stage/results/movement_unit_validation/','Historical penalty-and-knee structural diagnostics')],[3.05,4.01])
    p('The numerical environment used Python 3.10.12, NumPy 2.1.3, SciPy 1.15.2 and Matplotlib 3.10.1 in WSL Ubuntu. Numerical fitting uses one BLAS/OMP thread; the free-time control uses six worker processes. Reproduction needs the existing wrist trajectory file. Motion data and body-model assets must be obtained under their respective access conditions [4]; code availability does not imply redistribution of those assets.')
    code(['cd minimum-jerk-motion-approximation',
          'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m motion_decomposition.final_pipeline',
          'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m motion_decomposition.complete_mainline',
          'python3 -m motion_decomposition.export_mainline_goals',
          'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m motion_decomposition.timing_bound_sensitivity',
          'python3 -m motion_decomposition.audit_report_evidence',
          'python3 -m unittest discover -s tests -q'])
    p('Code repository: https://github.com/Hikaru-shih/minimum-jerk-motion-approximation. This local report does not assert that the latest third-stage files have already been pushed to the remote repository. The report builder reads saved evidence without refitting or altering historical outputs.')

    page('Appendix A Full count sweep and parameter inventory')
    table('Every unique partition under the nominal filter',['Window s','K','Position RMSE cm','Velocity RMSE m/s','Max position cm'],[
        (r['window'].replace('_','–'),r['K'],f"{100*r['position_rmse_m']:.3f}",f"{r['velocity_rmse_m_s']:.4f}",f"{100*r['max_position_error_m']:.3f}")
        for r in sorted(m['sweep'],key=lambda r:(r['window'],r['K']))],[.85,.4,1.85,1.85,2.11])
    p('The main detector setting is relative prominence 0.10, yielding K=4 for 18–21 s and K=3 for 30–33 s. All other rows are sensitivity partitions; they are not alternative nominal settings chosen after inspecting error. Boundary samples and the complete list of prominence values producing each partition are retained in summary.json.')
    table('Final main fitting settings',['Setting','Value'],[
        ('Solver and derivative','TRF; three-point finite differences; x_scale=jac'),('Objective','Raw XYZ position squared error only'),
        ('Termination tolerances','ftol=xtol=gtol=1e−12'),('Per-start evaluation budget','max_nfev=1000; finite-difference residual calls can exceed nfev'),
        ('Starts and seed','Four starts, seed 23'),('Endpoint conditions','Observed initial and final positions fixed'),
        ('Nominal local time fractions','Outward 0.45, inward 0.20'),('Initial outward extension','0.25 of the local boundary scale')],[2.2,4.86])

    page('Appendix B Exported cumulative goals and interval data')
    p('Cumulative goals are computed by adding fitted displacements in candidate-interval order to the observed initial crop position. They are not copied trajectory samples and, during overlap, need not be visited by the wrist. The initial position is not included in K. The final cumulative goal equals the observed final crop position by construction.')
    for label in ('18_21','30_33'):
        gg=[row for row in goals if row['window']==label]
        table(f'{label.replace("_","–")} s cumulative goals',['i','Goal X m','Goal Y m','Goal Z m'],[
            (g['component'],*[f'{float(g[f"goal_{c}_m"]):.6f}' for c in 'xyz']) for g in gg],[.45,2.2,2.2,2.21])
    table('Candidate boundaries and nominal free displacement rank',['Window s','Candidate boundaries s','Rank / free columns'],[
        (label.replace('_','–'),' / '.join(f'{v:.4f}' for v in m['details'][label]['detection']['knots']),f"{m['details'][label]['fit']['free_rank']} / {m['details'][label]['fit']['free_columns']}") for label in ('18_21','30_33')],[.85,4.71,1.5])
    p('The nominal endpoint-constrained free-displacement matrices have ranks 3/3 and 2/2. Full column rank supports a unique linear least-squares displacement solution for those fixed timings. It does not prove uniqueness of the nonlinear timing solution or the generating movement structure.')
    table('Checks used to freeze the report evidence',['Check','Verified count or result'],[
        ('Main comparison records',audit['mainline_comparisons']),('Main local-time runs',audit['mainline_runs']),
        ('Sensitivity conditions and runs',f"{audit['sensitivity_comparisons']} / {audit['sensitivity_runs']}"),
        ('Exported components / goals',audit['goals']),('Nominal sensitivity reproduced',str(audit['nominal_reproduced'])),('Numerical tests','54 passed')],[3.6,3.46])

    page('Appendix C References and source attribution')
    refs=[
        '[1] Sato, H., Mitake, H., and Hasegawa, S. (2025). Artist-Directable Motion Generation Using Overlapping Minimum Jerk Trajectories for Interactive Embodied Social Agent Adapting to Dynamic Environments. Computer Animation and Virtual Worlds, 36, e70075. https://doi.org/10.1002/cav.70075',
        '[2] Flash, T., and Hogan, N. (1985). The coordination of arm movements: An experimentally confirmed mathematical model. Journal of Neuroscience, 5(7), 1688–1703. https://doi.org/10.1523/JNEUROSCI.05-07-01688.1985',
        '[3] Liao, J. Y., and Kirsch, R. F. (2014). Characterizing and Predicting Submovements during Human Three-Dimensional Arm Reaches. PLOS ONE, 9(7), e103387. https://doi.org/10.1371/journal.pone.0103387',
        '[4] Mahmood, N., Ghorbani, N., Troje, N. F., Pons-Moll, G., and Black, M. J. (2019). AMASS: Archive of Motion Capture as Surface Shapes. Proceedings of ICCV, 5442–5451. https://amass.is.tue.mpg.de/',
        '[5] SciPy 1.15.2 documentation. scipy.optimize.least_squares. https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.optimize.least_squares.html',
        '[6] SciPy 1.15.2 documentation. scipy.signal.savgol_filter and scipy.signal.find_peaks. https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.signal.savgol_filter.html ; https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.signal.find_peaks.html']
    for ref in refs:p(ref)
    heading('Attribution of the implemented method',2)
    p('Reference [1] supplies the relevant overlapping motion-generation representation; [2] is the classical minimum-jerk movement reference; [3] motivates examining fitted submovements and cancellation. The fixed-endpoint elimination, speed-lobe count heuristic, local timing boxes, sensitivity settings and independent audit are the specific implementation choices of this project. They are not claimed as exact algorithms or validated parameters from those papers.')
    p('The report uses original experiment plots rather than reproducing published figures. Dataset identification is based on the stored project source and [4]. Numerical API descriptions are based on the version-specific official documentation [5–6]. The motivating research correspondence is not treated as a published algorithm or as justification for specific parameter values.')
    p('The earlier study notes include additional submovement and model-selection references. They are not listed as implemented methods here because the final main experiment does not reproduce them. Formal proof of physiological movement units, a universal goal-count criterion and a learned motion model remain outside the supported conclusions.')

    for container in (doc.styles.element,doc.element):
        for border in container.xpath('.//w:pBdr'):border.getparent().remove(border)
    doc.save(TARGET)
    manifest={'docx':str(TARGET),'figures':figures,'equations':len(equations),'tables':len(tables),
              'source_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in [ROOT/'results/mainline/summary.json',ROOT/'results/mainline/goals.csv',ROOT/'results/timing_bound_sensitivity/summary.json',ROOT/'results/timing_bound_sensitivity/audit.json',ROOT/'results/final_pipeline/summary.json',ROOT/'archive/third_stage/results/movement_unit_validation/summary.json']},
              'scope':'Editable comprehensive third-stage report, final main method plus explicitly historical diagnostics; no new fits'}
    (ASSETS/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(TARGET),'figures':len(figures),'equations':len(equations),'tables':len(tables)},indent=2))


if __name__=='__main__':main()
