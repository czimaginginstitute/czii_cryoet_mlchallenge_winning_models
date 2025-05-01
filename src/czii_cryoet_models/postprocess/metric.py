import numpy as np
import torch
from sklearn.metrics import roc_auc_score

"""
Derived from:
https://github.com/cellcanvas/album-catalog/blob/main/solutions/copick/compare-picks/solution.py
"""

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree


class ParticipantVisibleError(Exception):
    pass


def compute_metrics(reference_points, reference_radius, candidate_points, distance_multiplier, beta):
    print(f"Computing metrics with {len(reference_points)} reference points and {len(candidate_points)} candidate points")
    
    if len(reference_points) == 0:
        print("No reference points, returning default metrics for no reference points")
        return (np.inf, 0.0, 0.0, 0.0, 0, len(candidate_points), 0, 0.0, 0.0, 0, len(candidate_points), 0)
    
    if len(candidate_points) == 0:
        print("No candidate points, returning default metrics for no candidate points")
        return (np.inf, 0.0, 0.0, 0.0, len(reference_points), 0, 0, 0.0, 0.0, 0, 0, len(reference_points))
    
    ref_tree = cKDTree(reference_points)
    threshold = reference_radius * distance_multiplier
    distances, indices = ref_tree.query(candidate_points)
    
    valid_distances = distances[distances != np.inf]
    average_distance = np.mean(valid_distances) if valid_distances.size > 0 else np.inf
    
    matches_within_threshold = distances <= threshold
    
    tp = int(np.sum(matches_within_threshold))
    fp = int(len(candidate_points) - tp)
    fn = int(len(reference_points) - tp)
    
    precision = tp / len(candidate_points) if len(candidate_points) > 0 else 0
    recall = tp / len(reference_points) if len(reference_points) > 0 else 0
    fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0
    
    num_reference_particles = len(reference_points)
    num_candidate_particles = len(candidate_points)
    num_matched_particles = tp
    percent_matched_reference = (tp / num_reference_particles) * 100
    percent_matched_candidate = (tp / num_candidate_particles) * 100
    
    return (average_distance, precision, recall, fbeta, num_reference_particles, 
            num_candidate_particles, num_matched_particles, percent_matched_reference, 
            percent_matched_candidate, tp, fp, fn)



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


def score(all_results, weights, beta=4.0):
    # all_results = {}
    # for run in runs:
    #     run_name = run.name
    #     print(f"Processing run: {run_name}")
    #     results = process_run(run)
    #     all_results[run_name] = results

    micro_avg_results = {}
    aggregate_fbeta = 0.0    
    type_metrics = {}
    for run_results in all_results.values():
        for particle_type, metrics in run_results.items():
            weight = weights.get(particle_type, 1.0)
            
            if particle_type not in type_metrics:
                type_metrics[particle_type] = {
                    'total_tp': 0,
                    'total_fp': 0,
                    'total_fn': 0,
                    'total_reference_particles': 0,
                    'total_candidate_particles': 0
                }
            
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            type_metrics[particle_type]['total_tp'] += tp
            type_metrics[particle_type]['total_fp'] += fp
            type_metrics[particle_type]['total_fn'] += fn
            type_metrics[particle_type]['total_reference_particles'] += metrics['num_reference_particles']
            type_metrics[particle_type]['total_candidate_particles'] += metrics['num_candidate_particles']
    
    for particle_type, totals in type_metrics.items():
        weight = weights.get(particle_type, 1.0)
        tp = totals['total_tp']
        fp = totals['total_fp']
        fn = totals['total_fn']
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0
        
        micro_avg_results[particle_type] = {
            'precision': precision,
            'recall': recall,
            'f_beta_score': fbeta,
            'total_reference_particles': int(totals['total_reference_particles']),
            'total_candidate_particles': int(totals['total_candidate_particles']),
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
        aggregate_fbeta += fbeta * weight

    total_weight = sum(weights.values()) if weights else len(micro_avg_results.keys())

    weighted_fbeta_sum = 0.0
    for particle_type, metrics in micro_avg_results.items():
        weight = weights.get(particle_type, 1.0)
        fbeta = metrics['f_beta_score']
        weighted_fbeta_sum += fbeta * weight

    aggregate_fbeta = weighted_fbeta_sum / total_weight
    
    print(f"Aggregate F-beta Score: {aggregate_fbeta} (beta={beta})")
    print("Micro-averaged metrics across all runs per particle type:")
    for particle_type, metrics in micro_avg_results.items():
        print(f"Particle: {particle_type}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall: {metrics['recall']}")
        print(f"  F-beta Score: {metrics['f_beta_score']} (beta={beta})")
        print(f"  Total Reference Particles: {metrics['total_reference_particles']}")
        print(f"  Total Candidate Particles: {metrics['total_candidate_particles']}")
        print(f"  TP: {metrics['tp']}")
        print(f"  FP: {metrics['fp']}")
        print(f"  FN: {metrics['fn']}")

    return aggregate_fbeta