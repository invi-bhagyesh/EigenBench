

import re
import json
from itertools import combinations


def get_pairs(n):
    return list(combinations(range(n), 2))

"""
Comparisons format: <choice> mode:[scenario, judge, eval1, eval2, score]
                    <criterion> mode: [criterion, scenario, judge, eval1, eval2, score]
"""

def extract_comparisons_with_ties(data, include_scenario=True):
    """
    Takes in evaluations with ties and <choice> responses, 
    returns comparisons as a list of [scenario, judge, eval1, eval2, score],
    where score in {0,1,2} for cross entropy classification
    """
    comparisons = []
    data_cleaned = []

    error_count = 0
    no_number_count = 0
    no_match_count = 0

    for i, item in enumerate(data):
        response = item['judge response']
        eval1_response = item['eval1 response']
        eval2_response = item['eval2 response']
        eval1_reflection = item.get('eval1 reflection')
        eval2_reflection = item.get('eval2 reflection')

        fields = [response, eval1_response, eval2_response, eval1_reflection, eval2_reflection]
        if any(f is None for f in fields[:3]):
            continue

        if any(f and re.search(r"Error in \w+ API call", f) for f in fields if f is not None):
            error_count += 1
            continue

        m = re.search(f'<choice>(.)</choice>', response)

        if m:
            try:
                score = int(m.group(1))
                if include_scenario:
                    comparisons.append([item['scenario_index'], item['judge'], item['eval1'], item['eval2'], score])
                else:
                    comparisons.append([item['judge'], item['eval1'], item['eval2'], score])
                data_cleaned.append(item)
            except:
                no_number_count += 1
                continue
        else:
            no_match_count += 1
            continue

    print(f"Number of comparisons with an API call error: {error_count}")
    print(f"Number of judge responses without a <choice> match: {no_match_count}")
    print(f"Number of judge responses without a number in the <choice> match: {no_number_count}")

    return comparisons, data_cleaned


def extract_comparisons_with_ties_criteria(data, num_criteria):
    """
    Takes in evaluations with ties and <criterion> responses, 
    returns comparisons as a list of [criterion, scenario, judge, eval1, eval2, score],
    where score in {0,1,2} for cross entropy classification
    """
    comparisons = []
    data_cleaned = []

    none_count = 0
    error_count = 0
    no_number_count = 0
    no_match_count = 0
    other_number_count = 0

    for i, item in enumerate(data):
        matched_any = False
        
        response = item['judge response']
        eval1_response = item['eval1 response']
        eval2_response = item['eval2 response']
        eval1_reflection = item.get('eval1 reflection')
        eval2_reflection = item.get('eval2 reflection')

        if response == None or eval1_response == None or eval2_response == None:
            none_count += 1
            continue

        fields = [response, eval1_response, eval2_response, eval1_reflection, eval2_reflection]
        if any(f and re.search(r"Error in \w+ API call", f) for f in fields if f is not None):
            error_count += 1
            continue

        for j in range(1,num_criteria+1):
            m = re.search(
                rf'<criterion_{j}_choice>\s*([0-2])\s*</criterion_{j}_choice>',
                response,
                flags=re.DOTALL
            )

            if m:
                try:
                    score = int(m.group(1))

                    if score in [0,1,2]:
                        comparisons.append([j-1, item['scenario_index'], item['judge'], item['eval1'], item['eval2'], score])
                        matched_any = True
                    else:
                        other_number_count += 1
                except:
                    no_number_count += 1
                    continue
            else:
                no_match_count += 1
                continue

        if matched_any:
            data_cleaned.append(item)

    print(f"Number of comparisons with a null response: {none_count}")
    print(f"Number of comparisons with an API call error: {error_count}")
    print(f"Number of judge responses missing a specific <criterion> match: {no_match_count}")
    print(f"Number of judge responses missing a number in the <criterion> match: {no_number_count}")
    print(f"Number of judge responses with a non-0/1/2 number in the <criterion> match: {other_number_count}")

    print(f'\nTotal comparisons generated: {len(comparisons)}/{len(data) * num_criteria}')

    return comparisons, data_cleaned


def handle_inconsistencies_with_ties(comparisons):
    """
    Takes in comparisons in the form [scenario, judge, eval1, eval2, score] with ties,
    converts pairs of comparisons to ties if inconsistent, otherwise keeps original responses
    """
    scenarios = list(set([i[0] for i in comparisons]))
    num_models = len(set([i[1] for i in comparisons]))

    comparisons_new = []

    for l in scenarios:
        scenario_set = [i for i in comparisons if i[1] == l]

        for judge in range(num_models):
            judge_set = [i for i in scenario_set if i[2] == judge]

            if len(judge_set)==0:
                continue

            for eval1, eval2 in get_pairs(num_models):
                subset = [i for i in judge_set if (i[3] == eval1 and i[4] == eval2) or (i[4] == eval1 and i[3] == eval2)]

                if len(subset) == 2:
                    j,k = subset[0], subset[1]
                    if j[-1] == 0:
                        comparisons_new.append(j)
                    elif j[-1] != k[-1]:
                        comparisons_new.append(j)
                    else:
                        comparisons_new.append([l, judge, j[3], j[4], 0])

                    if k[-1] == 0:
                        comparisons_new.append(k)
                    elif j[-1] != k[-1]:
                        comparisons_new.append(k)
                    else:
                        comparisons_new.append([l, judge, k[3], k[4], 0])
                    
                elif len(subset) == 1:
                    comparisons_new.append(subset[0])

    return comparisons_new


def handle_inconsistencies_with_ties_criteria(comparisons):
    """
    Takes in comparisons in the form [criterion, scenario, judge, eval1, eval2, score] with ties,
    converts pairs of comparisons to ties if inconsistent, otherwise keeps original responses
    """
    num_criteria = len(set([i[0] for i in comparisons]))
    scenarios = list(set([i[1] for i in comparisons]))
    num_models = len(set([i[2] for i in comparisons ] + [i[3] for i in comparisons] + [i[4] for i in comparisons]))

    comparisons_new = []

    for c in range(num_criteria):
        criteria_set = [i for i in comparisons if i[0] == c]

        for l in scenarios:
            scenario_set = [i for i in criteria_set if i[1] == l]

            for judge in range(num_models):
                judge_set = [i for i in scenario_set if i[2] == judge]

                if len(judge_set)==0:
                    continue

                for eval1, eval2 in get_pairs(num_models):
                    subset = [i for i in judge_set if (i[3] == eval1 and i[4] == eval2) or (i[4] == eval1 and i[3] == eval2)]

                    if len(subset) == 2:
                        j,k = subset[0], subset[1]
                        if j[-1] == 0:
                            comparisons_new.append(j)
                        elif j[-1] != k[-1]:
                            comparisons_new.append(j)
                        else:
                            comparisons_new.append([c, l, judge, j[3], j[4], 0])

                        if k[-1] == 0:
                            comparisons_new.append(k)
                        elif j[-1] != k[-1]:
                            comparisons_new.append(k)
                        else:
                            comparisons_new.append([c, l, judge, k[3], k[4], 0])
                        
                    elif len(subset) == 1:
                        comparisons_new.append(subset[0])
    
    return comparisons_new


def load_evaluations(filepath):
    """Load evaluations from JSON or JSONL."""
    data = []
    if filepath.endswith('.jsonl'):
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        with open(filepath, 'r') as f:
            data.extend(json.load(f))
    return data
