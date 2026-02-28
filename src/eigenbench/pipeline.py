import torch
from torch.utils.data import DataLoader
from .data_utils import *
from .bt import *
from .eigentrust import *

"""
    # eigenbench pipeline: extract() and fit()
    from eigenbench import extract, fit
    comparisons, names = extract('evaluations.jsonl', num_criteria=8)
    results = fit(comparisons, num_models=len(names))
"""

def extract(filepath=None, num_criteria=None, collapse=True,
            # collect args (only used if filepath is None)
            criteria=None, scenarios=None, models=None,
            mode='criteria', allow_ties=True, partition_size=4,
            max_tokens=4096, multiturn=True, efficient=False,
            output_dir='transcript'):
    """
    Extract comparisons from evaluation data.
    If filepath is given, loads from file. Otherwise runs collect first.
    """
    if filepath is None:
        from .evaluation import run_evaluation
        run_evaluation(criteria, scenarios, models,
                       allow_ties=allow_ties, partition_size=partition_size,
                       max_tokens=max_tokens, multiturn=multiturn, mode=mode,
                       efficient=efficient, output_dir=output_dir)

        import os, glob
        files = glob.glob(f'{output_dir}/*/evaluations.jsonl')
        filepath = max(files, key=os.path.getmtime)
        print(f'Using: {filepath}')

    data = load_evaluations(filepath)
    print(f'Loaded {len(data)} evaluations')

    if num_criteria:
        comparisons, cleaned = extract_comparisons_with_ties_criteria(data, num_criteria)
        comparisons = handle_inconsistencies_with_ties_criteria(comparisons)
        if collapse:
            comparisons = [[0] + c[1:] for c in comparisons]
    else:
        comparisons, cleaned = extract_comparisons_with_ties(data)
        comparisons = handle_inconsistencies_with_ties(comparisons)

    names = sorted({n for item in cleaned for n in
                    [item.get('eval1_name',''), item.get('eval2_name',''), item.get('judge_name','')]
                    if n})

    print(f'{len(comparisons)} comparisons, {len(names)} models')
    return comparisons, names


def fit(comparisons, num_models, d=2, num_criteria=1,
        lr=1e-3, weight_decay=0, max_epochs=200,
        batch_size=32, alpha=0, device='cpu', save_path=None):
    """
    Train BTD model and compute EigenTrust scores.
    """
    dataset = Comparisons_criteria(comparisons)
    model = VectorBTD_criteria(num_criteria, num_models, d).to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history = train_vector_bt(model, loader, lr=lr, weight_decay=weight_decay,
                                   max_epochs=max_epochs, device=device,
                                   save_path=save_path, use_btd=True)

    T = compute_trust_matrix_ties(model, device=device)
    t = eigentrust(T, alpha=alpha)

    order = t.argsort(descending=True)
    rankings = [(idx.item(), t[idx].item()) for idx in order]

    return {'model': model, 'loss_history': loss_history,
            'trust_matrix': T, 'trust_scores': t, 'rankings': rankings}
