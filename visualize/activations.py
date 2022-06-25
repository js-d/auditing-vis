import torch
import torch.nn as nn
from collections import OrderedDict


class ModuleHook:
    def __init__(self, module: nn.Module):
        """`module` is a layer whose activations we want"""
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None  # will contain activations of layer of interest

    def hook_fn(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        """executes when applying `module` to `input`, with output `output`"""
        self.module = module
        self.features = output  # activations of the layer corresponding to module

    def close(self):
        self.hook.remove()


# Used to get activations for all layers
def hook_model(model: nn.Module, input: torch.Tensor):
    """
    Return a function `hook` that returns these activations for each layer.
    Define a dictionary mapping layer names to `ModuleHook` instances.
    After evaluating the model, the `feature` attribute of each instance contains the activations of the corresponding layer.
    `hook` then queries this dictionaries.


    Args:
        model (nn.Module):
        input (torch.Tensor):

    Returns:
        [type]: [description]
    """

    features = OrderedDict()

    # recursively initialize ModuleHook instance for every layer
    def hook_layers(net: nn.Module, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                # initialize ModuleHook instance
                features["_".join(prefix + [name])] = ModuleHook(layer)
                # recurse on submodules
                hook_layers(layer, prefix=prefix + [name])

    # for every layer features[layer].features will contain the activations of layer
    hook_layers(model)

    def hook(layer: str):
        """returns activations of layer
        to be used after evaluating model on batch
        (so that features[layer].features is not None)"""
        if layer == "input":
            out = input
        elif layer == "labels":
            out = list(features.values())[-1].features
        else:
            assert (
                layer in features
            ), f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
            out = features[layer].features  # most important line
        assert (
            out is not None
        ), "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
        return out

    return hook


def build_layer_dict(model: nn.Module):
    """Returns a dictionary mapping layer names to the associated `nn.Module` objects

    Args:
        model (nn.Module):

    Returns:
        layer_dict: dict
    """
    layer_dict = {}

    def update_dict(net: nn.Module, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                layer_dict["_".join(prefix + [name])] = layer
                update_dict(layer, prefix=prefix + [name])

    update_dict(model)
    return layer_dict


def single_layer_acts(model: nn.Module, batch: torch.Tensor, layer_name: str):
    """
    Return the activations in layer `layer` of `model`, given input `batch`
    Args:
        model (nn.Module):
        input (torch.Tensor):
        layer_name (str):

    Returns:
        torch.Tensor:
    """
    layer_dict = build_layer_dict(model)
    acts_obs = ModuleHook(layer_dict[layer_name])
    _ = model(batch)
    return acts_obs.features
