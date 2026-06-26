from time import time

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import random


def conf_interval(data, confidence_level=0.95):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    z = stats.norm.ppf(0.975)

    half_width = z * std / np.sqrt(len(data))

    return (
        mean - half_width,
        mean + half_width
    )

#### Primary Task

def simulate_patient_flow(bed_distribution, Ward_A_LOS, Ward_B_LOS, Ward_C_LOS, total_beds, simulation_days=365):
    beds_A, beds_B = bed_distribution
    beds_C = total_beds - beds_A - beds_B 

    if beds_A + beds_B > total_beds:
        raise ValueError("The sum of beds in Ward A and Ward B exceeds the total number of beds.")

    arrivals_A_total = 0
    arrivals_B_total = 0
    arrivals_C_total = 0

    Wards_A = []
    Wards_B = []
    Wards_C = []

    admitted_A = 0
    admitted_B = 0
    admitted_C = 0

    rejected_A = 0
    rejected_B = 0
    rejected_C = 0

    overflow_B = 0

    occupancy_A = []
    occupancy_B = []
    occupancy_C = []

    Ward_A_arrival_rate = lambda t: max(0, -(1/3650)*t**2 + (1/10)*t)
    Ward_B_arrival_rate = lambda t: 0.2 * Ward_A_arrival_rate(t)
    Ward_C_arrival_rate = lambda t: 6

 
    for day in range(simulation_days):

        # Discharges
        Wards_A = [los-1 for los in Wards_A if los-1 > 0]
        Wards_B = [los-1 for los in Wards_B if los-1 > 0]
        Wards_C = [los-1 for los in Wards_C if los-1 > 0]

        # Arrivals
        arrivals_A = np.random.poisson(Ward_A_arrival_rate(day))
        arrivals_B = np.random.poisson(Ward_B_arrival_rate(day))
        arrivals_C = np.random.poisson(Ward_C_arrival_rate(day))

        # Regular patients
        arrivals_A_total += arrivals_A
        for _ in range(arrivals_A):
            if len(Wards_A) < beds_A:
                Wards_A.append(np.ceil(Ward_A_LOS()))
                admitted_A += 1
            else:
                rejected_A += 1

        # Intensive care patients
        arrivals_B_total += arrivals_B
        for _ in range(arrivals_B):
            if len(Wards_B) < beds_B:
                Wards_B.append(np.ceil(Ward_B_LOS()))
                admitted_B += 1
            
            elif len(Wards_A) < beds_A:
                Wards_A.append(np.ceil(Ward_B_LOS()))
                admitted_B += 1
                overflow_B += 1

            else:
                rejected_B += 1

        arrivals_C_total += arrivals_C
        for _ in range(arrivals_C):
            if len(Wards_C) < beds_C:
                Wards_C.append(np.ceil(Ward_C_LOS()))
                admitted_C += 1
            else:
                rejected_C += 1

        occupancy_A.append(len(Wards_A))
        occupancy_B.append(len(Wards_B))
        occupancy_C.append(len(Wards_C))

    return {
        'admitted': {
            'A': admitted_A,
            'B': admitted_B,
            'C': admitted_C
        },

        'rejected': {
            'A': rejected_A,
            'B': rejected_B,
            'C': rejected_C,
            'Sum': rejected_A + rejected_B + rejected_C
        },

        'overflow_B': overflow_B,

        'blocking_probability': {
            'A': rejected_A / arrivals_A_total if arrivals_A_total else 0,
            'B': rejected_B / arrivals_B_total if arrivals_B_total else 0,
            'C': rejected_C / arrivals_C_total if arrivals_C_total else 0
        },

        'mean_fraction_occupied': {
            'A': np.mean(np.array(occupancy_A) / beds_A),
            'B': np.mean(np.array(occupancy_B) / beds_B),
            'C': np.mean(np.array(occupancy_C) / beds_C)
        },

        'Utilsation': {
            'A': np.mean(np.array(occupancy_A) / beds_A),
            'B': np.mean(np.array(occupancy_B) / beds_B),
            'C': np.mean(np.array(occupancy_C) / beds_C)
        }
    }


#### Sensitivity Analysis


def find_optimal_bed_distribution(
    total_beds,
    Ward_A_LOS,
    Ward_B_LOS,
    Ward_C_LOS,
    simulation_days=365,
    stage1_replications=5,
    stage2_replications=100,
    top_k=10,
    min_A=15,
    max_A=45,
    min_B=3,
    max_B=20
):

    print("Finding optimal bed distribution...")
    print("Stage 1: Screening candidate allocations")

    stage1_seeds = np.arange(stage1_replications)
    stage2_seeds = np.arange(stage2_replications)

    candidate_allocations = []

    total_allocations = 0
    for beds_A in range(min_A, max_A + 1):
        for beds_B in range(min_B, max_B + 1):

            beds_C = total_beds - beds_A - beds_B

            if beds_C >= 1:
                total_allocations += 1

    completed = 0

    for beds_A in range(min_A, max_A + 1):

        for beds_B in range(min_B, max_B + 1):

            beds_C = total_beds - beds_A - beds_B

            if beds_C < 1:
                continue

            rejections = []

            for seed in stage1_seeds:
                np.random.seed(seed)
                result = simulate_patient_flow(
                    (beds_A, beds_B),
                    Ward_A_LOS=Ward_A_LOS,
                    Ward_B_LOS=Ward_B_LOS,
                    Ward_C_LOS=Ward_C_LOS,
                    total_beds=total_beds,
                    simulation_days=simulation_days
                )

                rejections.append(
                    result["rejected"]["Sum"]
                )

            candidate_allocations.append({
                "beds_A": beds_A,
                "beds_B": beds_B,
                "beds_C": beds_C,
                "mean_rejections": np.mean(rejections)
            })


    best_candidates = candidate_allocations[:top_k]
    results_table = []

    best_distribution = None
    best_mean_rejections = float("inf")

    for idx, candidate in enumerate(best_candidates, start=1):

        beds_A = candidate["beds_A"]
        beds_B = candidate["beds_B"]
        beds_C = candidate["beds_C"]

        rejections = []
        overflows = []
        util_A = []
        util_B = []
        util_C = []

        for seed in stage2_seeds:

            np.random.seed(seed)

            result = simulate_patient_flow(
                (beds_A, beds_B),
                Ward_A_LOS=Ward_A_LOS,
                Ward_B_LOS=Ward_B_LOS,
                Ward_C_LOS=Ward_C_LOS,
                total_beds=total_beds,
                simulation_days=simulation_days
            )

            rejections.append(
                result["rejected"]["Sum"]
            )

            overflows.append(
                result["overflow_B"]
            )

            util_A.append(
                result["mean_fraction_occupied"]["A"]
            )

            util_B.append(
                result["mean_fraction_occupied"]["B"]
            )

            util_C.append(
                result["mean_fraction_occupied"]["C"]
            )

        mean_rejections = np.mean(rejections)

        results_table.append({
            "beds_A": beds_A,
            "beds_B": beds_B,
            "beds_C": beds_C,
            "mean_rejections": mean_rejections,
            "std_rejections": np.std(rejections, ddof=1),
            "mean_overflows": np.mean(overflows),
            "mean_util_A": np.mean(util_A),
            "mean_util_B": np.mean(util_B),
            "mean_util_C": np.mean(util_C)
        })

        if mean_rejections < best_mean_rejections:

            best_mean_rejections = mean_rejections

            best_distribution = (
                beds_A,
                beds_B,
                beds_C
            )

    results_table.sort(
        key=lambda x: x["mean_rejections"]
    )

    return (
        best_distribution,
        best_mean_rejections,
        results_table
    )

def rejection_method(total_beds,Ward_A_LOS,Ward_B_LOS,Ward_C_LOS,simulation_days=365,replications=100):

    candidate_allocations = []

    for beds_A in range(1, total_beds):
        for beds_B in range(1, total_beds - beds_A):

            beds_C = total_beds - beds_A - beds_B

            if beds_C < 1:
                continue

            rejections = []

            for _ in range(10):    
                result = simulate_patient_flow(...)
                rejections.append(result['rejected']['Sum'])

            candidate_allocations.append(
                (
                    np.mean(rejections),
                    beds_A,
                    beds_B,
                    beds_C
                )
            )

    candidate_allocations.sort(key=lambda x: x[0])
    return candidate_allocations[0]

def sensitivity_analysis(bed_distributions, Ward_A_LOS, Ward_B_LOS, Ward_C_LOS, total_beds, simulation_days=365):
    results = {}
    for beds_A, beds_B in bed_distributions:
        result = simulate_patient_flow((beds_A, beds_B), Ward_A_LOS, Ward_B_LOS, Ward_C_LOS, total_beds, simulation_days)
        results[(beds_A, beds_B)] = result
    return results

