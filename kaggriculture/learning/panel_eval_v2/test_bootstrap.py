"""Synthetic pre-import entry checks; no actual panel is started."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from . import bootstrap
from . import experiment


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'project'
        for name in bootstrap.NAMESPACES:
            (self.root/name).mkdir(parents=True, exist_ok=True)
        for name in bootstrap.INITIALIZERS:
            target = self.root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bootstrap.ROOT/name, target)
        self.script = self.root/'learning'/'panel_eval_v2'/'bootstrap.py'
        shutil.copyfile(bootstrap.ROOT/'learning'/'panel_eval_v2'/'bootstrap.py', self.script)
        for name in ('runner', 'worker', 'independent_worker'):
            (self.script.parent/(name+'.py')).write_text(
                'import sys\nprint("synthetic-entry:"+sys.argv[1])\n', encoding='utf-8')

    def command(self, module, *, flags=True):
        prefix = Path(self.temp.name)/'fresh-cache'
        startup = ['-B','-X','pycache_prefix='+str(prefix),'-I'] if flags else ['-B']
        return [sys.executable, *startup, str(self.script), module, 'expected-argument']

    def test_real_initializer_manifest_matches_frozen_reference(self):
        reference = experiment._reference_freeze()
        frozen = {**reference['runtime_files'], **reference['new_source_sha256']}
        expected = {name:digest for name,digest in frozen.items() if name.endswith('/__init__.py')}
        self.assertEqual(bootstrap.INITIALIZERS, expected)
        files = set(frozen) | set(experiment.SUPPLEMENTAL_FILES)
        dirs = {Path(name).parent.as_posix() for name in files
                if name.startswith('learning/') and name.endswith('.py')}
        expected_namespaces = {name for name in dirs if not (bootstrap.ROOT/name/'__init__.py').exists()}
        self.assertEqual(set(bootstrap.NAMESPACES), expected_namespaces)
        bootstrap.verify_initializers()

    def test_all_fixed_entries_receive_arguments_from_direct_script(self):
        for module in sorted(bootstrap.MODULES):
            with self.subTest(module=module):
                result = subprocess.run(self.command(module), capture_output=True, text=True, check=False)
                self.assertEqual((result.returncode, result.stdout.strip(), result.stderr),
                                 (0,'synthetic-entry:expected-argument',''))
        self.assertFalse((Path(self.temp.name)/'fresh-cache').exists())

    def test_modified_or_added_initializer_fails_before_entry(self):
        target = self.root/'learning'/'panel_eval_v2'/'__init__.py'
        target.write_text('raise RuntimeError("should not execute")', encoding='utf-8')
        result = subprocess.run(self.command('learning.panel_eval_v2.worker'), capture_output=True, check=False)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual((result.stdout,result.stderr),(b'',b''))
        target.unlink()
        sourceless = target.with_suffix('.pyc')
        sourceless.write_bytes(b'synthetic')
        result = subprocess.run(self.command('learning.panel_eval_v2.worker'), capture_output=True, check=False)
        self.assertNotEqual(result.returncode,0)
        sourceless.unlink()
        frozen = self.root/'learning'/'bc_crop'/'__init__.py'
        frozen.write_bytes(frozen.read_bytes()+b'\n')
        result = subprocess.run(self.command('learning.panel_eval_v2.worker'), capture_output=True, check=False)
        self.assertNotEqual(result.returncode,0)

    def test_missing_flags_or_unlisted_module_fails(self):
        for command in (self.command('learning.panel_eval_v2.worker', flags=False),
                        self.command('learning.panel_eval_v2.unlisted')):
            result = subprocess.run(command, capture_output=True, check=False)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual((result.stdout,result.stderr),(b'',b''))

    def test_later_regular_package_cannot_replace_project_namespace(self):
        external = Path(self.temp.name)/'external'
        package = external/'learning'
        package.mkdir(parents=True)
        marker = external/'wrong-package-executed'
        (package/'__init__.py').write_text('open('+repr(str(marker))+',"w").write("bad")\n', encoding='utf-8')
        probe = ('import runpy,sys; sys.path.append(sys.argv[1]); '
                 'sys.argv=[sys.argv[2],"learning.panel_eval_v2.worker","expected-argument"]; '
                 'runpy.run_path(sys.argv[0],run_name="__main__")')
        prefix = Path(self.temp.name)/'fresh-cache'
        command = [sys.executable,'-B','-X','pycache_prefix='+str(prefix),'-I','-c',probe,str(external),str(self.script)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual((result.returncode,result.stdout.strip(),result.stderr),
                         (0,'synthetic-entry:expected-argument',''))
        self.assertFalse(marker.exists())

    def test_later_namespace_cannot_supply_project_submodules(self):
        external = Path(self.temp.name)/'external'
        package = external/'learning'
        package.mkdir(parents=True)
        marker = external/'wrong-submodule-executed'
        (package/'unexpected.py').write_text('open('+repr(str(marker))+',"w").write("bad")\n', encoding='utf-8')
        (self.script.parent/'worker.py').write_text(
            'try:\n import learning.unexpected\nexcept ModuleNotFoundError:\n print("isolated")\n'
            'else:\n print("unexpected")\n', encoding='utf-8')
        probe = ('import runpy,sys; sys.path.append(sys.argv[1]); '
                 'sys.argv=[sys.argv[2],"learning.panel_eval_v2.worker","expected-argument"]; '
                 'runpy.run_path(sys.argv[0],run_name="__main__")')
        prefix = Path(self.temp.name)/'fresh-cache'
        command = [sys.executable,'-B','-X','pycache_prefix='+str(prefix),'-I','-c',probe,str(external),str(self.script)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual((result.returncode,result.stdout.strip(),result.stderr),(0,'isolated',''))
        self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()
