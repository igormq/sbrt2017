from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import sys
import os

import logging
import logging.config
import yaml

import numpy as np
from scipy import sparse

import inspect
import yaml

from .hparams import HParams

import re

logger = logging.getLogger(__name__)


def safe_mkdirs(path):
    ''' Safe makedirs
    Directory is created with command `makedir -p`.
    Returns:
        `path` if the directory already exists or is created
    Exception:
        OSError if something is wrong
    '''
    try:
        os.makedirs(path)
    except OSError, e:
        if e.errno != 17:  # 17 = file exists
            raise

    return path


def get_from_module(module, name, params=None, regex=False):
    """ Get a class or method from a module given its name
    """
    members = inspect_module(module, regex=regex)

    if name is None or name.lower() == 'none':
        return None

    members = {k.lower().strip(): v for k, v in members.items()}

    try:
        member = members[name.lower().strip()]
        # is a class and must be instantiate if params is not none
        if (member and params is not None) and inspect.isclass(member):
            return member(**HParams().parse(params).values())

        return member
    except KeyError, e:
        raise KeyError("%s not found in %s.\n Valid values are: %s" %
                       (name, module, ', '.join(members.keys())))


def inspect_module(module, to_dict=True, regex=False):
    modules = {}
    if regex:
        pattern = re.compile(module)
        for key, value in sys.modules.items():
            if pattern.match(key):
                modules[key] = value
    else:
        modules = {module: sys.modules[module]}

    members = []
    for key, value in modules.items():
        members.extend(inspect.getmembers(value, lambda member:
                                          hasattr(member, '__module__') and
                                          member.__module__ == key))

    if to_dict:
        return dict(members)

    return members


def ld2dl(ld):
    '''Transform a list of dictionaries in a dictionaries with lists
    # Note
        All dictionaries have the same keys
    '''
    return dict(zip(ld[0], zip(*[d.values() for d in ld])))

def check_ext(fname, ext):
    # Adding dot
    ext = ext if ext[0] == '.' else '.' + ext
    fname, f_ext = os.path.splitext(fname)

    if f_ext == ext:
        return True

    return False


def parse_nondefault_args(args, default_args):
    # removing default arguments
    args_default = {k: v for k, v in vars(default_args).items()
                    if k not in [arg.split('-')[-1] for arg in sys.argv
                                 if arg.startswith('-')]}
    args_nondefault = {k: v for k, v in vars(args).items()
                       if k not in args_default or args_default[k] != v}

    args_nondefault = HParams().parse(args_nondefault)

    return args_nondefault


def setup_logging(default_path='logging.yaml', default_level=logging.INFO,
                  env_key='LOG_CFG'):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
