from .builder import STOPPERS


@STOPPERS.register_module()
class DictionaryStopper(dict):
    pass
