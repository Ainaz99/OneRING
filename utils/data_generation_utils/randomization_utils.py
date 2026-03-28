import random
from typing import Sequence, List, Dict, Any, Union, Set

from allenact.utils.misc_utils import unzip


def weighted_random_permutation(l: Sequence, weights: Sequence[float]) -> List:
    ind_to_weight = {i: w for i, w in enumerate(weights)}

    permuted_inds = []
    for _ in range(len(l)):
        subinds, subweights = unzip(list(ind_to_weight.items()), 2)
        permuted_inds.append(
            random.choices(
                subinds,
                weights=subweights,
                k=1,
            )[0]
        )
        del ind_to_weight[permuted_inds[-1]]

    return [l[i] for i in permuted_inds]


def weighted_random_permutation_from_counts(
    l: Union[Sequence, Set], counts: Union[Sequence[float], Dict[Any, float]]
) -> List:
    if isinstance(l, Set):
        assert isinstance(counts, Dict), "If l is a set, counts must be a dict."
        l = list(l)

    if isinstance(counts, Dict):
        counts = [counts[ll] for ll in l]

    return weighted_random_permutation(l=l, weights=[1.0 / (c + 1e-8) for c in counts])
