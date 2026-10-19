ABLATION_VERSION = 1

ABLATIONS = {
    'full': {'router_mode': 'learned', 'expert_map': 'full', 'router_map': 'full'},
    'uniform': {'router_mode': 'uniform', 'expert_map': 'full', 'router_map': 'full'},
    'static': {'router_mode': 'static', 'expert_map': 'full', 'router_map': 'full'},
    'expert_mean': {'router_mode': 'learned', 'expert_map': 'spatial_mean', 'router_map': 'full'},
    'router_mean': {'router_mode': 'learned', 'expert_map': 'full', 'router_map': 'spatial_mean'},
    'both_mean': {'router_mode': 'learned', 'expert_map': 'spatial_mean', 'router_map': 'spatial_mean'},
}


def configure_ablation(args, checkpoint = False):
    """Resolve an experiment and validate its saved meaning before reconstruction."""
    fields = ('ablation', 'ablation_version', 'ablation_config')
    present = [hasattr(args, field) for field in fields]
    if checkpoint and any(present) and not all(present):
        raise ValueError('Checkpoint has incomplete ablation metadata')

    name = getattr(args, 'ablation', 'full')
    if not isinstance(name, str) or name not in ABLATIONS:
        raise ValueError(f'Unknown ablation: {name!r}')

    version = getattr(args, 'ablation_version', ABLATION_VERSION)
    if type(version) is not int or version != ABLATION_VERSION:
        raise ValueError(f'Unsupported ablation version: {version!r}')

    config = dict(ABLATIONS[name])
    if hasattr(args, 'ablation_config') and args.ablation_config != config:
        raise ValueError('Saved ablation configuration differs from the selected experiment')

    # Checkpoints predating ablation metadata use the unchanged full model.
    args.ablation = name
    args.ablation_version = ABLATION_VERSION
    args.ablation_config = config

    return config


def select_interaction_map(interaction_map, mode):
    """Keep the complete map or broadcast each text token's visual-position mean."""
    if mode == 'full':
        return interaction_map

    if mode == 'spatial_mean':
        return interaction_map.mean(dim = -1, keepdim = True).expand_as(interaction_map)

    raise ValueError(f'Unsupported interaction-map mode: {mode!r}')
