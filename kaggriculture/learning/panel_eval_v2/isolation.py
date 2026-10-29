"""Validate a process-local empty bytecode-cache namespace before use.

``-B`` suppresses writes but does not stop Python from reading existing pyc
files.  A fresh ``-X pycache_prefix`` redirects lookup away from project caches.
The caller must exclusively create and mark the scratch directory, keep it
alive until every child exits, and clean it afterwards.  This check does not
prove scratch ownership or prevent a same-user race after validation.  A
trusted launcher must use these flags *before* importing project code: checking
inside a process cannot undo bytecode already read during its own startup.
The prefix redirects CPython's normal source-cache lookup; it does not reject
explicit sourceless bytecode, custom import loaders, or already loaded modules.
"""
import os
from pathlib import Path
import re
import stat
import sys
import tempfile


_MODULE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z')
_PREFIX_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z')
_DEVICES = frozenset(('CON','PRN','AUX','NUL',
    *('COM'+str(n) for n in range(1,10)), *('LPT'+str(n) for n in range(1,10))))


def _absolute(value):
    try:
        supplied = Path(value)
        if not supplied.is_absolute() or '..' in supplied.parts:
            raise ValueError('Expected an absolute non-traversing path')
        if os.name == 'nt':
            for part in supplied.parts:
                if part != supplied.anchor and (':' in part or part.rstrip(' .') != part or Path(part).is_reserved()):
                    raise ValueError('Unsupported Windows isolation path')
        return Path(os.path.abspath(supplied))
    except (TypeError, OSError):
        raise ValueError('Invalid isolation path') from None


def _reject_reparse(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        raise ValueError('Isolation path cannot be inspected') from None
    if (stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400
        or getattr(Path(path), 'is_junction', lambda: False)()):
        raise ValueError('Isolation path contains a link or reparse point')
    return True


def _check_ancestors(path):
    for item in (path, *path.parents):
        _reject_reparse(item)


def _inside(path, root):
    return path == root or path.is_relative_to(root)


def validate_prefix(owned_root, prefix, project_root):
    """Validate a still-absent direct child of an existing OS-temp directory.

    The child name must use ASCII letters, digits, hyphens or underscores and
    cannot be a Windows device name.  This is a static check; the caller is
    responsible for exclusive ownership and must not create the prefix while
    the associated process is running.
    """
    try:
        supplied_name = Path(prefix).name
    except TypeError:
        raise ValueError('Invalid isolation path') from None
    if not _PREFIX_NAME.fullmatch(supplied_name) or supplied_name.upper() in _DEVICES:
        raise ValueError('Invalid isolation prefix name')
    owned, prefix, project = (_absolute(value) for value in (owned_root, prefix, project_root))
    temp = _absolute(tempfile.gettempdir())
    for path in (temp, owned, prefix, project):
        _check_ancestors(path)
    if (temp == Path(temp.anchor) or not temp.is_dir() or temp.resolve(strict=True) != temp
        or not owned.is_dir() or owned.resolve(strict=True) != owned
        or not project.is_dir() or project.resolve(strict=True) != project
        or owned == temp or not owned.is_relative_to(temp)
        or prefix.parent != owned or not _PREFIX_NAME.fullmatch(prefix.name) or prefix.name.upper() in _DEVICES
        or _inside(project, owned) or _inside(owned, project)
        or _reject_reparse(prefix)):
        raise ValueError('Untrusted or overlapping isolation paths')
    # A dangling link is rejected by lstat above even though exists() is false.
    try:
        prefix.lstat()
    except FileNotFoundError:
        return prefix
    except OSError:
        raise ValueError('Isolation prefix cannot be inspected') from None
    raise ValueError('Isolation prefix must not exist')


def validate_runtime(owned_root, prefix, project_root):
    """Require startup flags and an absent isolated prefix for this process.

    Invoke again after the protected work, before removing the owned scratch.
    A process that imported this module from untrusted prior cache is already
    too late; the parent launcher must supply the flags on process creation.
    """
    expected = validate_prefix(owned_root, prefix, project_root)
    argv = getattr(sys, 'orig_argv', ())
    if (not isinstance(argv, list) or len(argv) < 4 or argv[1:3] != ['-B', '-X']
        or not isinstance(argv[3], str) or not argv[3].startswith('pycache_prefix=')
        or sys.flags.dont_write_bytecode != 1 or sys.dont_write_bytecode is not True
        or sys.flags.optimize != 0 or 'PYTHONOPTIMIZE' in os.environ
        or not isinstance(sys._xoptions.get('pycache_prefix'), str)
        or not isinstance(sys.pycache_prefix, str)):
        raise ValueError('Required isolated Python startup flags are absent')
    startup_value = argv[3].split('=', 1)[1]
    if (_absolute(startup_value) != expected
        or _absolute(sys._xoptions['pycache_prefix']) != expected
        or _absolute(sys.pycache_prefix) != expected):
        raise ValueError('Python cache prefix does not match the isolated path')
    validate_prefix(owned_root, expected, project_root)


def module_command(executable, owned_root, prefix, project_root, module):
    """Construct the argument prefix; append only separately validated inputs.

    This helper never takes or embeds a game seed.  The returned list is meant
    for a direct process invocation, not an interpolated shell command.
    """
    target = validate_prefix(owned_root, prefix, project_root)
    binary = _absolute(executable)
    _check_ancestors(binary)
    if (not binary.is_file() or not isinstance(module, str) or len(module) > 256
        or not _MODULE.fullmatch(module)):
        raise ValueError('Invalid isolated module command')
    return [str(binary), '-B', '-X', 'pycache_prefix=' + str(target), '-m', module]


def bootstrap_command(executable, owned_root, prefix, project_root, module):
    """Start a fixed entry through the direct script before project imports."""
    allowed = ('learning.panel_eval_v2.runner', 'learning.panel_eval_v2.worker',
               'learning.panel_eval_v2.independent_worker')
    if module not in allowed:
        raise ValueError('Unexpected panel entry')
    base = module_command(executable, owned_root, prefix, project_root, module)
    script = _absolute(project_root)/'learning'/'panel_eval_v2'/'bootstrap.py'
    _check_ancestors(script)
    if not script.is_file():
        raise ValueError('Panel bootstrap unavailable')
    return base[:4] + ['-I', str(script), module]
