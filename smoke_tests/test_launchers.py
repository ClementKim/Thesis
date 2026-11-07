"""Exercise shell launcher routing without running training or evaluation."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('full', 'spectral_only', 'evidence_only', 'random_selection', 'direct_image')


class LauncherSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix = 'ipmoe-launcher-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.bin = self.directory / 'bin'
        self.bin.mkdir()
        self.log = self.directory / 'launches.jsonl'
        self.recorder = self.directory / 'record.py'
        self.recorder.write_text(
            'import json, os, sys\n'
            'record = {"argv": sys.argv[1:], "env": {key: os.environ.get(key) '
            'for key in ("IPMOE_CHECKPOINT", "PYTHONHASHSEED")}}\n'
            'with open(os.environ["SMOKE_LOG"], "a") as output:\n'
            '    output.write(json.dumps(record) + "\\n")\n')
        executable = self.bin / 'torchrun'
        executable.write_text('#!/bin/sh\nexec "$SMOKE_PYTHON" "$SMOKE_RECORDER" "$@"\n')
        executable.chmod(0o755)

    def launch(self, script, arguments = (), environment = None):
        env = os.environ.copy()
        for key in ('SEEDS', 'LLM_TYPE', 'ABLATION', 'CHECKPOINT_DIR', 'OUT', 'DATASETS', 'JUDGE',
                    'IPMOE_CHECKPOINT', 'PYTHONHASHSEED'):
            env.pop(key, None)
        env.update({'PATH': str(self.bin) + os.pathsep + os.environ.get('PATH', os.defpath),
                    'SMOKE_PYTHON': sys.executable, 'SMOKE_RECORDER': str(self.recorder),
                    'SMOKE_LOG': str(self.log), 'SEEDS': '42', 'DATASETS': 'POPE', 'JUDGE': 'exact_matching'})
        if environment:
            env.update(environment)
        self.log.write_text('')

        result = subprocess.run(['bash', str(ROOT / script), *arguments], cwd = self.directory,
                                env = env, capture_output = True, text = True, timeout = 15)
        records = [json.loads(line) for line in self.log.read_text().splitlines()]

        return result, records

    def option(self, record, name):
        arguments = record['argv']
        values = [arguments[index + 1] if argument == name else argument.split('=', 1)[1]
                  for index, argument in enumerate(arguments) if argument == name or argument.startswith(name + '=')]
        self.assertTrue(values, f'{name} missing from {arguments}')
        return values[-1]

    def variant_directory(self, mode):
        return 'svd_evidence_prefix' + ('' if mode == 'full' else '/' + mode)

    def assert_training_launch(self, result, records, mode, checkpoint_dir = None):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record['argv'][:3], ['--standalone', '--nproc-per-node=gpu', 'main.py'])
        self.assertEqual(self.option(record, '--ablation'), mode)
        self.assertEqual(self.option(record, '--seed'), '42')
        self.assertEqual(record['env']['PYTHONHASHSEED'], '42')
        checkpoint_dir = checkpoint_dir or 'checkpoint/' + self.variant_directory(mode)
        output_dir = str(Path(checkpoint_dir) / 'seed_42')
        self.assertEqual(self.option(record, '--output-dir'), output_dir)
        self.assertTrue((self.directory / output_dir / 'train.log').is_file())

    def test_training_default_and_environment_variants_use_separate_paths(self):
        result, records = self.launch('run.sh')
        self.assert_training_launch(result, records, 'full')
        self.assertEqual(self.option(records[0], '--llm-type'), 'llama')
        for mode in VARIANTS:
            with self.subTest(mode = mode):
                self.assert_training_launch(*self.launch('run.sh', environment = {'ABLATION': mode}), mode)

    def test_training_cli_overrides_environment_and_preserves_custom_directory(self):
        for mode in VARIANTS:
            for arguments in (['--ablation', mode], ['--ablation=' + mode]):
                with self.subTest(mode = mode, arguments = arguments):
                    environment = {'ABLATION': 'random_selection' if mode == 'full' else 'full'}
                    self.assert_training_launch(*self.launch('run.sh', arguments, environment), mode)

        custom = str(self.directory / 'custom checkpoints')
        result, records = self.launch('run.sh', ['--ablation', 'evidence_only'],
                                      {'ABLATION': 'spectral_only', 'CHECKPOINT_DIR': custom, 'LLM_TYPE': 'qwen'})
        self.assert_training_launch(result, records, 'evidence_only', custom)
        self.assertEqual(self.option(records[0], '--llm-type'), 'qwen')

    def test_evaluation_loads_matching_checkpoint_and_output_directory(self):
        cases = [({}, 'full')]
        cases.extend(({'ABLATION': mode}, mode) for mode in VARIANTS)
        cases.append(({'ABLATION': 'random_selection', 'CHECKPOINT_DIR': str(self.directory / 'custom checkpoints'),
                       'OUT': str(self.directory / 'custom results')}, 'random_selection'))
        for environment, mode in cases:
            with self.subTest(environment = environment):
                directory = self.variant_directory(mode)
                checkpoint_dir = environment.get('CHECKPOINT_DIR', 'checkpoint/' + directory)
                work_dir = str(Path(environment.get('OUT', 'evaluation/' + directory)) / 'seed_42')
                checkpoint = str(Path(checkpoint_dir) / 'seed_42' / 'checkpoint.pt')
                path = self.directory / checkpoint
                path.parent.mkdir(parents = True, exist_ok = True)
                path.touch()

                result, records = self.launch('eval.sh', environment = environment)

                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(len(records), 1)
                record = records[0]
                self.assertEqual(record['argv'][:3], ['--standalone', '--nproc-per-node=gpu', 'eval.py'])
                self.assertEqual(record['env'], {'IPMOE_CHECKPOINT': checkpoint, 'PYTHONHASHSEED': '42'})
                self.assertEqual(self.option(record, '--model'), 'IPMoE')
                self.assertEqual(self.option(record, '--data'), 'POPE')
                self.assertEqual(self.option(record, '--judge'), 'exact_matching')
                self.assertEqual(self.option(record, '--work-dir'), work_dir)
                self.assertNotIn('--ablation', record['argv'])
                self.assertTrue((self.directory / work_dir / 'eval.log').is_file())

    def test_invalid_or_missing_ablation_is_rejected_before_launch(self):
        cases = [('run.sh', [], {'ABLATION': 'invalid'}, 'Unknown ablation'),
                 ('run.sh', ['--ablation', 'invalid'], {}, 'Unknown ablation'),
                 ('run.sh', ['--ablation=invalid'], {}, 'Unknown ablation'),
                 ('run.sh', ['--ablation'], {}, '--ablation requires a value'),
                 ('run.sh', ['--ablation='], {}, 'Unknown ablation'),
                 ('eval.sh', [], {'ABLATION': 'invalid'}, 'Unknown ablation'),
                 ('eval.sh', ['--ablation', 'spectral_only'], {}, 'Set ABLATION'),
                 ('eval.sh', ['--ablation=evidence_only'], {}, 'Set ABLATION'),
                 ('eval.sh', ['--ablation'], {}, 'Set ABLATION')]
        for script, arguments, environment, message in cases:
            with self.subTest(script = script, arguments = arguments, environment = environment):
                result, records = self.launch(script, arguments, environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(records, [])
                self.assertIn(message, result.stderr)


if __name__ == '__main__':
    unittest.main()
