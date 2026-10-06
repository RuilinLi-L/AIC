"""Frozen runtime policies for V20 GPU admission and allocator limits."""
from copy import deepcopy

EXCLUSIVE = 'exclusive_70gib'
SHARED = 'gpu3_shared_measured_v1'


def gpu_policy(name=EXCLUSIVE):
    if name == EXCLUSIVE:
        return {'name': name, 'gpu_order': [0, 1, 2, 3], 'shared_gpus': [],
                'minimum_free_mib': 71680, 'limits': {}}
    if name != SHARED:
        raise ValueError('unknown V20 GPU runtime policy')
    return deepcopy({'name': name, 'gpu_order': [3, 0, 1, 2], 'shared_gpus': [3],
        'minimum_free_mib': 36864,
        'limits': {'loraplus_rank32': {'cap_gib': 32, 'minimum_free_mib': 36864},
                   'dora_rank32': {'cap_gib': 48, 'minimum_free_mib': 53248},
                   'rank32_control': {'cap_gib': 32, 'minimum_free_mib': 36864},
                   'rank32_dropout': {'cap_gib': 32, 'minimum_free_mib': 36864},
                   'resolution384_rank32': {'cap_gib': 48, 'minimum_free_mib': 53248}}})


def task_limit(policy, label):
    if policy['name'] == EXCLUSIVE:
        return {'cap_gib': None, 'minimum_free_mib': 71680}
    for recipe, limit in policy['limits'].items():
        if label.startswith(recipe):
            return dict(limit)
    raise ValueError('shared GPU task must identify its frozen recipe')
