# -*- coding: utf-8 -*-
# Copyright (c) Percy.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Data Privacy optimization using Reinforcement Learning Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from .__version__ import __title__, __description__, __version__
from .__version__ import __author__, __author_email__, __license__
from .__version__ import __copyright__

from .dprl import Dprl
from .exceptions import (
    UnImpException, FlagError, ParamError, StepWarning
)