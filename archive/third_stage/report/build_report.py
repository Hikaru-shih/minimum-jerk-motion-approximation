import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent
BASE = ROOT/'results/uniform_baseline_comparison'


def main():
    comparison = json.loads((BASE/'summary.json').read_text())
    review = json.loads((ROOT/'results/candidate_review/summary.json').read_text())
    doc = Document()
    section = doc.sections[0]
    section.page_width,section.page_height = Inches(8.5),Inches(11)
    section.top_margin,section.bottom_margin = Inches(.65),Inches(.65)
    section.left_margin,section.right_margin = Inches(.75),Inches(.75)
    for name in ('Normal','Title','Subtitle','Heading 1','Heading 2','Caption'):
        style = doc.styles[name]
        style.font.name = 'Times New Roman'; style.font.color.rgb = RGBColor(0,0,0)
        style.font.underline = False
        style.font.size = Pt(11)
        fonts = style.element.get_or_add_rPr().find(qn('w:rFonts'))
        if fonts is not None:
            for attr in ('asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme'):
                fonts.attrib.pop(qn('w:'+attr),None)
            fonts.set(qn('w:ascii'),'Times New Roman'); fonts.set(qn('w:hAnsi'),'Times New Roman')
    doc.styles['Normal'].paragraph_format.space_after = Pt(6)
    doc.styles['Normal'].paragraph_format.line_spacing = 1.08
    doc.styles['Title'].font.size = Pt(18)
    doc.styles['Heading 1'].font.size = Pt(13)
    doc.styles['Heading 1'].font.bold = True
    doc.styles['Heading 2'].font.size = Pt(11)
    doc.styles['Heading 2'].font.bold = True
    for name in ('Heading 1','Heading 2'):
        doc.styles[name].paragraph_format.space_before = Pt(10)
        doc.styles[name].paragraph_format.space_after = Pt(6)
    doc.styles['Caption'].font.size = Pt(9)
    doc.styles['Caption'].font.italic = False
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'),'PAGE'); footer._p.append(field)
    doc.core_properties.author = 'Pin-Kuang Shih'
    doc.core_properties.title = 'Fitting Wrist Motion with Overlapping Minimum Jerk Displacements'

    def para(text): return doc.add_paragraph(text)
    def heading(text): return doc.add_heading(text,level=1)
    def page(title): doc.add_page_break(); heading(title)
    def table(headers,rows,widths):
        t = doc.add_table(rows=1,cols=len(headers)); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
        for column,width in zip(t.columns,widths): column.width = Inches(width)
        for cell,text in zip(t.rows[0].cells,headers): cell.text = text
        for row in rows:
            for cell,text in zip(t.add_row().cells,row): cell.text = str(text)
        for row_index,row in enumerate(t.rows):
            trpr = row._tr.get_or_add_trPr()
            if row_index==0: trpr.append(OxmlElement('w:tblHeader'))
            trpr.append(OxmlElement('w:cantSplit'))
            for c,cell in enumerate(row.cells):
                cell.width = Inches(widths[c]); cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                props = cell._tc.get_or_add_tcPr()
                borders = OxmlElement('w:tcBorders')
                for edge in ('top','left','bottom','right'):
                    e = OxmlElement('w:'+edge); e.set(qn('w:val'),'single'); e.set(qn('w:sz'),'4'); e.set(qn('w:color'),'D9D9D9'); borders.append(e)
                props.append(borders)
                margins = OxmlElement('w:tcMar')
                for edge in ('top','bottom','left','right'):
                    e = OxmlElement('w:'+edge); e.set(qn('w:w'),'70'); e.set(qn('w:type'),'dxa'); margins.append(e)
                props.append(margins)
                if row_index==0:
                    shade = OxmlElement('w:shd'); shade.set(qn('w:fill'),'E7EDF3'); props.append(shade)
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.0
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c==0 else WD_ALIGN_PARAGRAPH.CENTER
                    for run in p.runs: run.font.size = Pt(10); run.bold = row_index==0
        doc.add_paragraph().paragraph_format.space_after = Pt(0)
        return t
    def figure(filename,caption,width=6.9):
        p = doc.add_paragraph(); p.paragraph_format.keep_with_next = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(BASE/'report_figures'/filename),width=Inches(width))
        doc.add_paragraph(caption,style='Caption')
    def row(window,k,method): return next(r for r in comparison['rows'] if r['window']==window and r['K']==k and r['method']==method)

    doc.add_paragraph('Fitting Wrist Motion with Overlapping Minimum Jerk Displacements',style='Title')
    para('Goal count selection and comparison with uniform timing')
    para('Pin-Kuang Shih')
    heading('Summary')
    para('I fitted displacement amplitudes and start and end times instead of copying intermediate goals from the observed trajectory. I compared goal counts from 1 to 12 and used a normalized cost-curve knee as a provisional rule for choosing a compact approximation. For two windows from the same wrist-motion clip, the rule selected three and four goals when the tested upper count was 8, 10, or 12.')
    para('At those counts, position and velocity errors were lower than both uniform-timing baselines evaluated at the same count. The first decomposition was consistent across the ten saved initializations; the second retained two different decompositions. These results support a limited fitting and count-selection prototype, rather than a uniquely identified set of human movement units.')
    heading('Data and scope')
    para('The data are the right-wrist trajectory extracted with SMPL+H from ACCAD/AMASS clip Female1Gestures_c3d/D3 - Conversation Gestures_poses.npz. I analyzed 18-21 s and 30-33 s at 120 Hz, with 361 samples per window. Positions are in global XYZ coordinates in meters. Each window is reconstructed independently, with its observed initial and final positions fixed.')
    para('This report evaluates two selected windows, not the complete clip or full-body motion. The count K denotes displacement components and cumulative goals; the initial position is stored separately.')
    heading('Research question')
    para('Can fitting both timing and displacement improve on fixed intervals, and can an explicit count rule choose a compact approximation without specifying an arbitrary position-error threshold? The count criterion and its limitations are evaluated separately from the quality of the fitted components.')

    page('Method and comparison protocol')
    para('Each component contributes a three-dimensional displacement multiplied by a minimum-jerk progress function [3]. The progress rises from zero to one between its fitted activation and completion times. Components add and may overlap; a new component does not wait for another to finish. Cumulative goals are obtained by adding successive displacement vectors to the initial position [1].')
    para('For fixed times and no opposition penalty, I solve an endpoint-constrained linear least-squares problem for the displacement vectors. With the custom opposition penalty, I optimize timing and displacement together. The final displacement is eliminated using the fixed-endpoint equality, leaving 5K-3 trajectory parameters.')
    para('The objective is mean squared XYZ position error plus lambda times a scaled penalty on simultaneous negative velocity dot products. I use lambda = 0.000001, length scale 0.3 m, minimum component duration 2/120 s, and a per-fit limit of 1000 function evaluations. This is my defined position-based objective, inspired by the decomposition problem in [2]; it does not reproduce that paper\'s velocity objective or penalty formula.')
    para('For each pair, the penalty takes the positive part of the negative velocity dot product, scales it by the squared window duration divided by the length scale, and squares it. It sums over pairs and averages over frames. Eligible fits must terminate successfully, remain finite, and satisfy the fixed endpoints.')
    para('For each K, I retain the lowest-objective eligible solution among ten timing starts: five uniform starts and five starts based on speed minima within time slots, including their perturbations. Timing remains free after initialization. Solver success does not establish a global optimum. The ten starts for K=1 are identical.')
    para('Count selection normalizes K and the fitted objective over the tested range, connects the endpoints of the curve, and chooses the interior point with maximum positive distance below that chord. A nonmonotone or incomplete curve is not silently smoothed. This adapts the knee idea in [2] to the present objective.')
    table(['Comparison','Timing','Intermediate positions'],[
        ['Uniform sampled','Equal intervals','Observed positions at interval ends'],
        ['Uniform fitted','Same equal intervals','Fitted displacement with the same penalty'],
        ['Fitted timing','Optimized start and end times','Fitted jointly with timing']], [1.35,1.8,3.75])
    para('All methods share raw targets, observed position endpoints, K, reconstruction model, and finite-difference evaluation. Uniform intervals extend on each internal side by one quarter of an interval. Off-grid sampled goals use linear interpolation. At K=6, intervals are 0.5 s and neighboring components overlap for 0.25 s. These are crop-based reproductions of the earlier recipe, not crops of its full-clip reconstruction.')
    para('The fixed-time fitted control has one linear-least-squares initialization; the adaptive search has ten timing starts. The comparison controls data, count, and objective, but not total compute. Improvements cannot be attributed to timing alone independently of the search procedure.')

    page('Errors at equal goal counts')
    para('Table 1 compares the candidate counts. Position and maximum errors are XYZ vector errors; velocity is computed with the same unfiltered finite difference for all trajectories. All 24 new fixed-time displacement controls returned eligible solutions.')
    rows=[]
    for window,k in (('18_21',3),('30_33',4)):
        for method,label in (('uniform_sampled_goals','Uniform sampled'),('uniform_fitted_displacements','Uniform fitted'),('adaptive_timing','Fitted timing')):
            r = row(window,k,method)
            rows.append([window.replace('_','-')+' s',str(k),label,f'{100*r["position_rmse_m"]:.3f}',f'{r["velocity_rmse_m_s"]:.4f}',f'{100*r["max_position_error_m"]:.3f}'])
    table(['Window','K','Method','Pos cm','Vel m/s','Max cm'],rows,[.8,.35,1.75,.85,1,.95])
    figure('errors_vs_count.png','Figure 1. Position and velocity RMSE at equal K. Dotted lines mark the candidate counts. The curve is evaluated on the fitted samples, not held-out motion.',6.9)
    para('The fitted-timing results have lower position and velocity RMSE than both uniform baselines at every tested K in these two windows. Uniform-grid errors are not necessarily monotone as K changes because moving the grid changes its alignment with the motion. Fitting position also does not guarantee lower velocity error.')

    page('First window trajectory comparison')
    figure('trajectories_18_21.png','Figure 2. First window, K=3 for every method. Left: XYZ position. Right: XYZ velocity. The count is held fixed so a larger number of goals cannot explain the improvement.',6.9)
    para('Relative to uniform timing with fitted displacement, position RMSE decreases from 3.837 cm to 1.414 cm, and velocity RMSE from 0.2393 m/s to 0.1223 m/s. The main Y and Z changes are better timed. Early X motion and some peaks and late details remain underrepresented. Maximum position error is 2.467 cm at about 20.317 s.')
    para('The three components overlap. A broad positive-X and positive-Z contribution combines with a shorter negative-Y and negative-Z contribution, followed by a negative-X and positive-Y contribution. These are geometric descriptions of fitted contributions, not labels of human intentions.')

    page('Second window trajectory comparison')
    figure('trajectories_30_33.png','Figure 3. Second window, K=4 for every method. The same endpoints and raw samples are used. Short velocity spikes are shown rather than removed from the evaluation.',6.9)
    para('Relative to uniform timing with fitted displacement, position RMSE decreases from 3.686 cm to 0.618 cm, and velocity RMSE from 0.3063 m/s to 0.1310 m/s. The major rise, descent, and return are better represented. Slow early drift and short velocity fluctuations remain; maximum position error is 1.570 cm at about 31.442 s.')
    para('The overlapping components separate a positive-Z contribution, a mainly negative-Z contribution, a mainly negative-Y contribution, and a later positive-Y return. This decomposition is one selected numerical explanation; the initialization analysis below shows an alternative arrangement.')

    page('Count selection and decomposition checks')
    table(['Tested upper K','First window selected K','Second window selected K'],[
        ['6','3','2'],['8','3','4'],['10','3','4'],['12','3','4']],[1.5,2.65,2.65])
    para('Extending the range matters: the second window\'s earlier K=2 choice changed to K=4. Agreement at upper limits 8, 10, and 12 provides finite-range support for the candidate counts, not invariance to every possible upper limit.')
    para('The candidates trade accuracy for fewer components. At K=12, position RMSE is 0.179 cm and 0.154 cm; velocity RMSE is 0.0658 m/s and 0.1227 m/s. The knee rule does not claim that later gains vanish. A provisional independent-Gaussian information-score check continued to prefer the upper limit; correlated residuals and degenerate components prevent treating it as a validated stopping rule.')
    table(['Window and component','Start s','End s','Displacement XYZ m'],[
        [c['window'].replace('_','-')+' C'+str(c['component']),f'{c["activation_s"]:.3f}',f'{c["completion_s"]:.3f}',f'({c["dx_m"]:+.3f}, {c["dy_m"]:+.3f}, {c["dz_m"]:+.3f})']
        for c in review['components']], [1.35,.8,.8,3.85])
    para('After matching component velocity signatures over all permutations, the first window\'s ten initializations agree numerically: the largest completion-time difference is below 0.000001 s. In the second window, seven runs approach the selected solution, while runs 0, 3, and 4 have position RMSE about 0.691 cm instead of 0.618 cm. Relative to the selected fit, matched completion times differ by up to 1.142 s and displacement vectors by up to 0.331 m. These are deterministic optimization probes, not an estimated 70 percent population success rate.')
    para('Cumulative intermediate goals can lie away from the wrist path during overlap. In the first window, the first two goals are about 29.8 cm and 12.3 cm from the nearest saved reconstructed samples. This is allowed by the additive representation; it does not verify an executable full-body pose sequence. Both candidates still contain simultaneous opposing velocity contributions.')

    page('Interpretation and limitations')
    para('The experiment now fits both goal displacements and timing, and compares the resulting count-quality tradeoff with clearly defined uniform-timing baselines. It provides candidate counts of three and four under a reproducible knee rule. It does not establish unique human movement units or an accuracy requirement under which those counts are optimal.')
    para('The count rule depends on the tested range, objective, penalty scale, and finite optimization budget. Only two selected windows from one clip were evaluated. The small penalty was a provisional controlled condition, not a universally tuned value. In the K=7-12 extension, all 120 penalized fits were eligible, but 37 of 120 unpenalized fits reached the evaluation cap; missing eligible solutions were not replaced by their last iterates.')
    para('Observed endpoints are fixed as a research assumption. Cropping does not make the original endpoint velocities zero, whereas all components are inactive or completed at the crop boundaries. The objective fits position; velocity is an evaluation measure. Without a validated noise model, short finite-difference spikes cannot all be classified as noise or meaningful movements.')
    para('Speed minima provide motion-related initialization clues, but overlapping components can merge speed peaks and small perturbations can add peaks. I therefore do not equate the observed number of speed lobes with the final count. The opposition penalty controls a numerical failure mode; it does not identify movement boundaries or prove biomechanical feasibility.')
    heading('Conclusion')
    para('Fitted timing and displacement improve the two tested windows relative to uniform timing at the same goal count. The cost-curve rule produces compact candidates that are consistent across the tested upper limits of 8, 10, and 12. The first candidate is stable within the saved starts; the second remains nonunique. I report these as initial results with explicit limits, rather than a complete solution to motion-unit decomposition.')
    heading('References')
    for text in (
        '[1] Sato H, Mitake H, Hasegawa S. Artist-Directable Motion Generation Using Overlapping Minimum Jerk Trajectories for Interactive Embodied Social Agent Adapting to Dynamic Environments. Computer Animation and Virtual Worlds. 2025;36:e70075. doi:10.1002/cav.70075.',
        '[2] Liao JY, Kirsch RF. Characterizing and Predicting Submovements during Human Three-Dimensional Arm Reaches. PLOS ONE. 2014;9(7):e103387. doi:10.1371/journal.pone.0103387.',
        '[3] Flash T, Hogan N. The coordination of arm movements: An experimentally confirmed mathematical model. Journal of Neuroscience. 1985;5(7):1688-1703. doi:10.1523/JNEUROSCI.05-07-01688.1985.'):
        p=para(text); p.paragraph_format.space_after=Pt(4)
        for run in p.runs: run.font.size=Pt(9)
    para('Reproducibility package: research/motion_decomposition/. Repository: https://github.com/Hikaru-shih/minimum-jerk-motion-approximation. Numerical environment: Python 3.10.12, NumPy 2.1.3, SciPy 1.15.2, Matplotlib 3.10.1.')
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT/'fitting_and_goal_selection_report.docx'
    for tree in (doc.element,doc.styles.element):
        for border in tree.xpath('.//w:pBdr'):
            border.getparent().remove(border)
    doc.save(path)
    print(path)


if __name__=='__main__':
    main()
