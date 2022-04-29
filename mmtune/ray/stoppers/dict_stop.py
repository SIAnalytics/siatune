from .builder import STOPPER


@STOPPER.register_module()
class DictionaryStopper(dict):
    pass
