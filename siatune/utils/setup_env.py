# Copyright (c) SI-Analytics. All rights reserved.
import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in siatune into the registries.

    Args:
        init_default_scope (bool): Whether initialize the siatune default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `siatune`, and all registries will build modules from siatune's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import siatune.codebase  # noqa: F401,F403
    import siatune.core  # noqa: F401,F403
    import siatune.tune  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('siatune')
        if never_created:
            DefaultScope.get_instance('siatune', scope_name='siatune')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'siatune':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "siatune", '
                          '`register_all_modules` will force the current'
                          'default scope to be "siatune". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'siatune-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='siatune')
