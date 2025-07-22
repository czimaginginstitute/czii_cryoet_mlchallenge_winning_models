"""
Derived from:
https://github.com/cellcanvas/album-catalog/blob/main/solutions/copick/compare-picks/solution.py
"""

import numpy as np
import pandas as pd

#from scipy.spatial import cKDTree
from scipy.spatial import KDTree
from tqdm import tqdm
from czii_cryoet_models.postprocess.utils import get_final_submission


class ParticipantVisibleError(Exception):
    pass


def compute_metrics(reference_points, reference_radius, candidate_points):
    num_reference_particles = len(reference_points)
    num_candidate_particles = len(candidate_points)

    if len(reference_points) == 0:
        return 0, num_candidate_particles, 0

    if len(candidate_points) == 0:
        return 0, 0, num_reference_particles

    ref_tree = KDTree(reference_points)
    candidate_tree = KDTree(candidate_points)
    raw_matches = candidate_tree.query_ball_tree(ref_tree, r=reference_radius)
    matches_within_threshold = []
    for match in raw_matches:
        matches_within_threshold.extend(match)
    # Prevent submitting multiple matches per particle.
    # This won't be be strictly correct in the (extremely rare) case where true particles
    # are very close to each other.
    matches_within_threshold = set(matches_within_threshold)
    tp = int(len(matches_within_threshold))
    fp = int(num_candidate_particles - tp)
    fn = int(num_reference_particles - tp)
    return tp, fp, fn


def score(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        row_id_column_name: str,
        distance_multiplier: float,
        beta: int,
        particle_radius: dict={},
        particle_weights: dict={},
        weighted=True,
) -> float:
    '''
    F_beta
      - a true positive occurs when
         - (a) the predicted location is within a threshold of the particle radius, and
         - (b) the correct `particle_type` is specified
      - raw results (TP, FP, FN) are aggregated across all experiments for each particle type
      - f_beta is calculated for each particle type
      - individual f_beta scores are weighted by particle type for final score
    '''
    particle_radius = {k: v * distance_multiplier for k, v in particle_radius.items()}

    # Filter submission to only contain experiments found in the solution split
    split_experiments = set(solution['experiment'].unique())
    submission = submission.loc[submission['experiment'].isin(split_experiments)]

    # Only allow known particle types
    if not set(submission['particle_type'].unique()).issubset(set(particle_weights.keys())):
        raise ParticipantVisibleError('Unrecognized `particle_type`.')

    dupes = solution.duplicated(subset=['experiment', 'x', 'y', 'z'], keep=False)
    if dupes.any():
        solution = solution.drop_duplicates(subset=['experiment', 'x', 'y', 'z']) # by default, keep first

    # Now ensure no duplicates remain
    assert solution.duplicated(subset=['experiment', 'x', 'y', 'z']).sum() == 0
    assert particle_radius.keys() == particle_weights.keys()
    
    results = {}
    for particle_type in particle_radius.keys():
        results[particle_type] = {
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0,
        }

    for experiment in split_experiments:
        for particle_type in solution['particle_type'].unique():
            reference_radius = particle_radius[particle_type]
            select = (solution['experiment'] == experiment) & (solution['particle_type'] == particle_type)
            reference_points = solution.loc[select, ['x', 'y', 'z']].values

            select = (submission['experiment'] == experiment) & (submission['particle_type'] == particle_type)
            candidate_points = submission.loc[select, ['x', 'y', 'z']].values

            if len(reference_points) == 0:
                reference_points = np.array([])
                reference_radius = 1

            if len(candidate_points) == 0:
                candidate_points = np.array([])

            tp, fp, fn = compute_metrics(reference_points, reference_radius, candidate_points)

            results[particle_type]['total_tp'] += tp
            results[particle_type]['total_fp'] += fp
            results[particle_type]['total_fn'] += fn

    #print(f'results: {results}')
    
    fbetas = []
    fbeta_weights = []
    particle_types = []
    for particle_type, totals in results.items():
        tp = totals['total_tp']
        fp = totals['total_fp']
        fn = totals['total_fn']

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0
        fbetas += [fbeta]
        fbeta_weights += [particle_weights.get(particle_type, 1.0)]
        particle_types += [particle_type]
        
    if weighted:
        aggregate_fbeta = np.average(fbetas,weights=fbeta_weights)
    else:
        aggregate_fbeta = np.mean(fbetas)
    
    return aggregate_fbeta, dict(zip(particle_types,fbetas))



def process_run(reference_picks, candidate_picks, pickable_objects, distance_multiplier=0.5, beta=4.0):    
    results = {}
    for particle_type in pickable_objects.keys():
        if particle_type in reference_picks:
            reference_points = reference_picks[particle_type]['points']
            reference_radius = reference_picks[particle_type]['radius']
        else:
            reference_points = np.array([])
            reference_radius = 1  # default radius if not available

        if particle_type in candidate_picks:
            candidate_points = candidate_picks[particle_type]['points']
        else:
            candidate_points = np.array([])

        (avg_distance, precision, recall, fbeta, num_reference, num_candidate, num_matched, 
            percent_matched_ref, percent_matched_cand, tp, fp, fn) = compute_metrics(
            reference_points,
            reference_radius,
            candidate_points,
            distance_multiplier,
            beta
        )
        results[particle_type] = {
            'average_distance': avg_distance,
            'precision': precision,
            'recall': recall,
            'f_beta_score': fbeta,
            'num_reference_particles': num_reference,
            'num_candidate_particles': num_candidate,
            'num_matched_particles': num_matched,
            'percent_matched_reference': percent_matched_ref,
            'percent_matched_candidate': percent_matched_cand,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    return results



def calc_metric(
        pred_df: pd.DataFrame, 
        gt_df: pd.DataFrame, 
        score_thresholds: dict={},
        particle_radius: dict={},
        particle_weights: dict={},
        output_dir: str='./output/jobs/job_0') -> dict:
    
    solution = gt_df.copy()
    solution['id'] = range(len(solution))
    submission = pred_df.copy()
    submission['id'] = range(len(submission))
    best_ths = {k: v for k, v in score_thresholds.items() if v > 0}
    # Find the best probability thresholding for each class
    for p in score_thresholds.keys():
        if p not in best_ths:
            print(f'Finding the best threshold for {p}')
            sol0a = solution[solution['particle_type']==p].copy()
            sub0a = submission[submission['particle_type']==p].copy()
            scores = []
            ths = np.arange(0.05,0.5,0.005) # prevent over picks; can take a long time if two many picks 
            for c in tqdm(ths):
                scores += [score(
                            sol0a.copy(),
                            sub0a[sub0a['conf']>c].copy(),
                            row_id_column_name = 'id',
                            distance_multiplier=0.5,
                            beta=4,
                            particle_radius=particle_radius,
                            particle_weights = particle_weights,
                            weighted = False)[0]]
            best_th = ths[np.argmax(scores)]
            best_ths[p] = best_th
    
    print(f'Best score threshold values {best_ths}')
    submission_pp = get_final_submission(submission, score_thresholds=best_ths, output_dir=output_dir)
    score_pp, particle_scores = score(
        solution[solution['particle_type']!='beta-amylase'].copy(),
        submission_pp.copy(),
        row_id_column_name = 'id',
        distance_multiplier=0.5,
        beta=4,
        particle_radius=particle_radius,
        particle_weights = particle_weights,
        )
    
    #print(f'particle_scores: {particle_scores}')
    result = {'score_' + k: v for k,v in particle_scores.items()}
    result['score'] = score_pp
    print(result)

    return result, best_ths