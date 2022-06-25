import torch
from lucent.optvis import objectives


@objectives.wrap_objective()
def caricature_obj(layer, direction, batch=None, cossim_pow=0):
    """objective optimized in image space by the caricature method

    inspired from https://colab.research.google.com/github/greentfrapp/lucent-notebooks/blob/master/notebooks/feature_inversion.ipynb#scrollTo=CJDqo-NRvZxL
    """

    @objectives.handle_batch(batch)
    def inner(model):
        layer_t = model(layer).squeeze(0)
        assert layer_t.shape == direction.shape
        dot = (layer_t * direction).sum()
        mag = torch.sqrt(torch.sum(layer_t ** 2))
        cossim = dot / (1e-6 + mag)
        return -dot * cossim ** cossim_pow

    return inner
