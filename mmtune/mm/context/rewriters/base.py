# Copyright (c) SI-Analytics. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict


class BaseRewriter(metaclass=ABCMeta):
    """Base class of rewriters."""

    @abstractmethod
    def __call__(self, context: Dict) -> Dict:
        """Rewrite the context.

        Args:
            context (Dict): The context to be rewritten.

        Returns:
            Dict: The context after rewriting.
        """
        return context
