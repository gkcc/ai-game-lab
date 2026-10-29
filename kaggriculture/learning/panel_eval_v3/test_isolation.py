import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import unittest

from . import isolation


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.owned = Path(self.temp.name)
        self.project = Path(__file__).resolve().parents[2]
        self.prefix = self.owned/'fresh-cache'
        self.env = os.environ.copy()
        self.env.pop('PYTHONOPTIMIZE', None)
        self.env.pop('PYTHONPYCACHEPREFIX', None)
        self.env.pop('PYTHONDONTWRITEBYTECODE', None)

    def run_python(self, arguments):
        return subprocess.run([sys.executable, *arguments], cwd=self.project, env=self.env,
            capture_output=True, text=True, timeout=30, check=False)

    def validation_script(self, *, modify=''):
        return ('from learning.panel_eval_v3 import isolation\n' + modify +
            '\ntry:\n isolation.validate_runtime(' + repr(str(self.owned)) + ',' + repr(str(self.prefix)) +
            ',' + repr(str(self.project)) + ')\nexcept ValueError:\n print("rejected")\n'
            'else:\n print("accepted")\n')

    def test_b_alone_reads_stale_pyc_but_fresh_prefix_loads_source(self):
        source=self.owned/'synthetic_cache_probe.py'
        source.write_text("VALUE='old'\n",encoding='utf-8')
        before=source.stat()
        # Use the source-adjacent cache even when this test runner itself uses
        # an isolated pycache prefix.
        cache = source.parent/'__pycache__'/(source.stem+'.'+sys.implementation.cache_tag+'.pyc')
        cache.parent.mkdir()
        py_compile.compile(str(source), cfile=str(cache), doraise=True)
        source.write_text("VALUE='new'\n",encoding='utf-8')
        os.utime(source, ns=(before.st_atime_ns,before.st_mtime_ns))
        self.assertEqual(source.stat().st_size,before.st_size)
        program=('import sys\nsys.path.insert(0,'+repr(str(self.owned))+')\n'
                 'import synthetic_cache_probe\nprint(synthetic_cache_probe.VALUE)\n')
        old=self.run_python(['-B','-c',program])
        self.assertEqual((old.returncode,old.stdout.strip()),(0,'old'))
        fresh=self.run_python(['-B','-X','pycache_prefix='+str(self.prefix),'-c',program])
        self.assertEqual((fresh.returncode,fresh.stdout.strip()),(0,'new'))
        self.assertFalse(self.prefix.exists())

    def test_real_subprocess_accepts_startup_flags_and_keeps_prefix_absent(self):
        cmd=['-B','-X','pycache_prefix='+str(self.prefix),'-c',self.validation_script()]
        result=self.run_python(cmd)
        self.assertEqual((result.returncode,result.stdout.strip()),(0,'accepted'))
        self.assertFalse(self.prefix.exists())

    def test_real_subprocess_rejects_missing_prefix_or_bytecode_flag(self):
        result=self.run_python(['-B','-c',self.validation_script()])
        self.assertEqual((result.returncode,result.stdout.strip()),(0,'rejected'))
        result=self.run_python(['-X','pycache_prefix='+str(self.prefix),'-c',self.validation_script()])
        self.assertEqual((result.returncode,result.stdout.strip()),(0,'rejected'))

    def test_real_subprocess_rejects_optimization_and_mutated_write_flag(self):
        result=self.run_python(['-B','-X','pycache_prefix='+str(self.prefix),'-O','-c',self.validation_script()])
        self.assertEqual((result.returncode,result.stdout.strip()),(0,'rejected'))
        self.assertFalse(self.prefix.exists())
        result=self.run_python(['-B','-X','pycache_prefix='+str(self.prefix),'-c',
                                self.validation_script(modify='import sys; sys.dont_write_bytecode=False')])
        self.assertEqual((result.returncode,result.stdout.strip()),(0,'rejected'))

    def test_prefix_must_be_absent_direct_and_outside_project(self):
        self.assertEqual(isolation.validate_prefix(self.owned,self.prefix,self.project),self.prefix)
        self.prefix.mkdir()
        with self.assertRaises(ValueError): isolation.validate_prefix(self.owned,self.prefix,self.project)
        with self.assertRaises(ValueError): isolation.validate_prefix(self.owned,self.owned.parent/'outside',self.project)
        with self.assertRaises(ValueError): isolation.validate_prefix(self.project,self.project/'cache',self.project)
        with self.assertRaises(ValueError): isolation.validate_prefix(self.owned,self.owned/'..'/'escape',self.project)
        for name in ('cache.','cache:stream','CON','com1','LPT9'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                isolation.validate_prefix(self.owned,self.owned/name,self.project)
        if os.name == 'nt':
            alias = self.owned.with_name(self.owned.name+'.')
            with self.assertRaises(ValueError):
                isolation.validate_prefix(alias,alias/'cache',self.project)

    def test_command_contains_only_startup_flags_and_module(self):
        command=isolation.module_command(sys.executable,self.owned,self.prefix,self.project,
                                         'learning.panel_eval_v3.worker')
        self.assertEqual(command[1:3],['-B','-X'])
        self.assertEqual(command[3],'pycache_prefix='+str(self.prefix))
        self.assertEqual(command[4:],['-m','learning.panel_eval_v3.worker'])
        with self.assertRaises(ValueError):
            isolation.module_command(sys.executable,self.owned,self.prefix,self.project,'bad;module')

    def test_panel_entry_command_uses_direct_isolated_bootstrap(self):
        command = isolation.bootstrap_command(sys.executable,self.owned,self.prefix,self.project,
                                               'learning.panel_eval_v3.worker')
        self.assertEqual(command[1:3], ['-B','-X'])
        self.assertEqual(command[3], 'pycache_prefix='+str(self.prefix))
        self.assertEqual(command[4:7], ['-I', str(self.project/'learning'/'panel_eval_v3'/'bootstrap.py'),
                                        'learning.panel_eval_v3.worker'])
        with self.assertRaises(ValueError):
            isolation.bootstrap_command(sys.executable,self.owned,self.prefix,self.project,'learning.unlisted')

    @unittest.skipUnless(os.name == 'nt','Windows junction check')
    def test_junction_ancestor_is_rejected_and_target_preserved(self):
        target=self.owned/'target'; target.mkdir(); marker=target/'keep.txt'; marker.write_text('keep',encoding='utf-8')
        junction=self.owned/'linked'
        result=subprocess.run(['cmd','/d','/c','mklink','/J',str(junction),str(target)],
                              capture_output=True,check=False)
        if result.returncode: self.skipTest('Windows junction creation unavailable')
        try:
            with self.assertRaises(ValueError): isolation.validate_prefix(junction,junction/'cache',self.project)
        finally:
            if junction.is_junction(): junction.rmdir()
        self.assertEqual(marker.read_text(encoding='utf-8'),'keep')


if __name__ == '__main__':
    unittest.main()
