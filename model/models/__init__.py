from .CrossEncoder import get_model_cross_encoder
from .Siamese import get_model_siamese
from .MultiTask import get_model_multi_task
__all__ = ["get_model"]
def get_model(model_type, **kwargs):
    if model_type == 'cross_encoder':
        return get_model_cross_encoder(**kwargs)
    elif model_type == 'siamese':
        return get_model_siamese(**kwargs)
    elif model_type == 'multi_task':
        return get_model_multi_task(**kwargs)
    else:
        raise ValueError(f"No supported model type: {model_type}")
    return None