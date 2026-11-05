"""Check that the root layout reproduces the saved pre-migration numerical results."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'archive/third_stage'


def function_tree(path,name):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)


def main():
    checked={}
    for name in ('final_pipeline','mainline','timing_bound_sensitivity'):
        with np.load(OLD/'results'/name/'fits.npz') as before,np.load(ROOT/'results'/name/'fits.npz') as after:
            assert set(before.files)==set(after.files)
            maximum=0.
            for key in before.files:
                assert before[key].shape==after[key].shape
                np.testing.assert_allclose(before[key],after[key],atol=1e-10,rtol=1e-10)
                if before[key].size:maximum=max(maximum,float(np.max(np.abs(before[key]-after[key]))))
            checked[name]={'arrays':len(before.files),'maximum_absolute_difference':maximum}
    assert (OLD/'results/mainline/goals.csv').read_bytes()==(ROOT/'results/mainline/goals.csv').read_bytes()
    for filename,names in {
        'fixed_times.py':['design_matrix','fit_displacements','fit_fixed_endpoints'],
        'timing_fit.py':['decode_times','encode_times','fit_timing'],
        'valley_fit.py':['fit_valley_units'],
        'timing_bound_sensitivity.py':['fit_variant'],
        'metrics.py':['trajectory_errors'],
    }.items():
        for name in names:
            assert function_tree(OLD/'motion_decomposition'/filename,name)==function_tree(ROOT/'motion_decomposition'/filename,name)
    extracted=[('step3_speed_minima.py','motion_utils.py',['speed_profile','local_minimum_runs']),
               ('step4_diagnose_initialization.py','motion_utils.py',['component_velocities','integrate','composition_metrics']),
               ('step5_fit_real_window.py','io_utils.py',['serializable']),
               ('step6_explain_real_components.py','io_utils.py',['write_csv']),
               ('step1_known_motion.py','synthetic.py',['generate_motion'])]
    for old,new,names in extracted:
        for name in names:assert function_tree(OLD/'motion_decomposition'/old,name)==function_tree(ROOT/'motion_decomposition'/new,name)
    assert hashlib.sha256((ROOT/'data/wrist_trajectory.npz').read_bytes()).hexdigest()==hashlib.sha256((ROOT/'archive/initial_and_followup/results/wrist_trajectory.npz').read_bytes()).hexdigest()
    result={'numerical_results_preserved':True,'input_bytes_preserved':True,'goals_csv_bytes_preserved':True,
            'fitting_function_bodies_preserved':True,'extracted_utility_bodies_preserved':True,'comparisons':checked,
            'archived_versions':['archive/initial_and_followup','archive/third_stage'],
            'note':'Root outputs were regenerated using saved control fits and unchanged numerical methods; source/protocol hashes now describe the root layout. Archived code and original result files were not rewritten.'}
    (ROOT/'archive/layout_migration.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
