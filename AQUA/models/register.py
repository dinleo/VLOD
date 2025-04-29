import inspect
from functools import partial

class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + "(name={}, items={})".format(
            self._name, list(self._module_dict.keys())
        )
        return format_str

    def __len__(self):
        return len(self._module_dict)

    def get_model(self, args):
        build_func = self.get(args.modelname)
        assert build_func is not None
        model = build_func(args)
        return model

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def registe_with_name(self, module_name=None, force=False):
        return partial(self.register, module_name=module_name, force=force)

    def register(self, module_build_function, module_name=None, force=False):
        """Register a module build function.
        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isfunction(module_build_function):
            raise TypeError(
                "module_build_function must be a function, but got {}".format(
                    type(module_build_function)
                )
            )
        if module_name is None:
            module_name = module_build_function.__name__
        if not force and module_name in self._module_dict:
            raise KeyError("{} is already registered in {}".format(module_name, self.name))
        self._module_dict[module_name] = module_build_function

        return module_build_function

MODULE_BUILD_FUNCS = Registry("model build functions")

from models.aqua import build_aqua
from models.groundingdino import build_groundingdino
from models.stage import build_stage1

MODULE_BUILD_FUNCS.register(build_aqua, "aqua")
MODULE_BUILD_FUNCS.register(build_groundingdino, "groundingdino")
MODULE_BUILD_FUNCS.register(build_stage1, "stage1")
