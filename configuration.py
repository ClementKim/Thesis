import math


MODEL_DEFAULTS = {
    'architecture': 'svd_evidence_prefix',
    'interaction_type': 'bilinear',
    'embed_dim': 256,
    'patch_size': 16,
    'svd_rank': 8,
    'evidence_dim': 64,
    'position_dim': 16,
    'drop': 0.0,
    'selector_keep_ratio': 0.5,
    'image_keep_ratio': 0.5,
    'selector_dim': 64,
    'selector_temperature': 1.0,
}


def configure_model(args, checkpoint = False):
    """Validate the single supported architecture before constructing its modules."""
    if checkpoint:
        if getattr(args, 'architecture', None) != MODEL_DEFAULTS['architecture']:
            raise ValueError('Checkpoint is not a svd_evidence_prefix checkpoint; interaction-ViT and earlier architectures are incompatible')
        missing = [field for field in MODEL_DEFAULTS if not hasattr(args, field)]
        if missing:
            raise ValueError(f'Checkpoint has incomplete model configuration: {missing}')

    removed = ('ablation', 'ablation_version', 'ablation_config', 'text_selector', 'image_selector',
               'selector_num_heads', 'direct_interaction_prefix', 'num_aux_tokens',
               'interaction_depth', 'num_heads', 'mlp_ratio', 'attn_drop')
    if any(hasattr(args, field) for field in removed):
        raise ValueError('Ablations, optional selector switches, expert settings, and interaction-ViT options are no longer supported')

    config = {field: getattr(args, field, default) for field, default in MODEL_DEFAULTS.items()}
    if config['architecture'] != MODEL_DEFAULTS['architecture'] or config['interaction_type'] != 'bilinear':
        raise ValueError('Only the bilinear svd_evidence_prefix architecture is supported')

    for field in ('embed_dim', 'patch_size', 'svd_rank', 'evidence_dim', 'position_dim', 'selector_dim'):
        if type(config[field]) is not int or config[field] <= 0:
            raise ValueError(f'{field} must be a positive integer')
    if config['position_dim'] < 2:
        raise ValueError('position_dim must be at least 2 for image row and column coordinates')

    for field in ('drop', 'selector_keep_ratio', 'image_keep_ratio', 'selector_temperature'):
        value = config[field]
        if type(value) not in (int, float):
            raise TypeError(f'{field} must be a finite number')
        try:
            value = float(value)
        except OverflowError as exc:
            raise ValueError(f'{field} must be a finite number') from exc
        if not math.isfinite(value):
            raise ValueError(f'{field} must be a finite number')
        config[field] = value
    if not 0 <= config['drop'] < 1:
        raise ValueError('drop must be in [0, 1)')
    if not 0 < config['selector_keep_ratio'] <= 1 or not 0 < config['image_keep_ratio'] <= 1:
        raise ValueError('Selector keep ratios must be in (0, 1]')
    if config['selector_temperature'] <= 0:
        raise ValueError('selector_temperature must be positive')

    for field, value in config.items():
        setattr(args, field, value)
    return config
