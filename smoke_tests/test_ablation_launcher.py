"""Isolated ablation-wrapper checks with a recording-only torchrun executable."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('full', 'spectral_only', 'evidence_only', 'random_selection', 'direct_image')


class AblationLauncherSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix = 'ipmoe-ablation-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.project = self.directory / 'project with spaces'
        self.project.mkdir()
        self.working_directory = self.directory / 'other working directory'
        self.working_directory.mkdir()
        for script in ('ablation.sh', 'run.sh', 'eval.sh'):
            shutil.copy2(ROOT / script, self.project / script)
        self.bin = self.directory / 'bin'
        self.bin.mkdir()
        self.log = self.directory / 'launches.jsonl'
        self.recorder = self.directory / 'record.py'
        self.recorder.write_text(
            'import json, os, sys\n'
            'arguments = sys.argv[1:]\n'
            'record = {"argv": arguments, "cwd": os.getcwd(), "env": {key: os.environ.get(key) '
            'for key in ("IPMOE_CHECKPOINT", "PYTHONHASHSEED")}}\n'
            'with open(os.environ["SMOKE_LOG"], "a") as output:\n'
            '    output.write(json.dumps(record) + "\\n")\n'
            'modes = [arguments[index + 1] if argument == "--ablation" else argument.split("=", 1)[1] '
            'for index, argument in enumerate(arguments) if argument == "--ablation" or argument.startswith("--ablation=")]\n'
            'status = int(os.environ.get("SMOKE_EXIT_STATUS", "0"))\n'
            'sys.exit(status or (7 if modes and modes[-1] == os.environ.get("SMOKE_FAIL_VARIANT") else 0))\n')
        executable = self.bin / 'torchrun'
        executable.write_text('#!/bin/sh\nexec "$SMOKE_PYTHON" "$SMOKE_RECORDER" "$@"\n')
        executable.chmod(0o755)

    def launch(self, arguments = (), environment = None):
        env = os.environ.copy()
        for key in ('MODE', 'ABLATIONS', 'SEEDS', 'BATCH_SIZE', 'CHECKPOINT_ROOT', 'OUT_ROOT',
                    'CHECKPOINT_DIR', 'OUT', 'ABLATION', 'LLM_TYPE', 'DATASETS', 'JUDGE',
                    'IPMOE_CHECKPOINT', 'PYTHONHASHSEED', 'SMOKE_FAIL_VARIANT', 'SMOKE_EXIT_STATUS'):
            env.pop(key, None)
        env.update({'PATH': str(self.bin) + os.pathsep + os.environ.get('PATH', os.defpath),
                    'SMOKE_PYTHON': sys.executable, 'SMOKE_RECORDER': str(self.recorder), 'SMOKE_LOG': str(self.log)})
        if environment is not None:
            env.update(environment)
        self.log.write_text('')

        result = subprocess.run(['bash', str(self.project / 'ablation.sh'), *arguments], cwd = self.working_directory,
                                env = env, capture_output = True, text = True, timeout = 20)
        records = [json.loads(line) for line in self.log.read_text().splitlines()]

        return result, records

    def option(self, record, name):
        arguments = record['argv']
        values = [arguments[index + 1] if argument == name else argument.split('=', 1)[1]
                  for index, argument in enumerate(arguments) if argument == name or argument.startswith(name + '=')]
        self.assertTrue(values, f'{name} missing from {arguments}')
        return values[-1]

    def resolved(self, record, path):
        return (Path(record['cwd']) / path).resolve()

    def test_default_training_runs_all_variants_with_one_seed_and_batch_one(self):
        result, records = self.launch()

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([self.option(record, '--ablation') for record in records], list(VARIANTS))
        for mode, record in zip(VARIANTS, records):
            self.assertEqual(record['cwd'], str(self.project))
            self.assertEqual(record['argv'][:3], ['--standalone', '--nproc-per-node=gpu', 'main.py'])
            self.assertEqual(self.option(record, '--seed'), '42')
            self.assertEqual(record['env']['PYTHONHASHSEED'], '42')
            self.assertEqual(self.option(record, '--batch-size'), '1')
            output = self.resolved(record, self.option(record, '--output-dir'))
            self.assertEqual(output, self.project / 'checkpoint' / 'ablations' / mode / 'seed_42')
            self.assertTrue((output / 'train.log').is_file())

    def test_subset_multiseed_custom_roots_and_forwarded_arguments(self):
        checkpoint_root = self.directory / 'custom checkpoint root'
        arguments = ['--batch-size', '2', '--llm-type', 'qwen', '--attn-implementation', 'eager']
        environment = {'ABLATIONS': 'evidence_only spectral_only', 'SEEDS': '7 19', 'BATCH_SIZE': '3',
                       'CHECKPOINT_ROOT': str(checkpoint_root), 'OUT_ROOT': str(self.directory / 'custom output root')}

        result, records = self.launch(arguments, environment)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected = [('evidence_only', '7'), ('evidence_only', '19'), ('spectral_only', '7'), ('spectral_only', '19')]
        self.assertEqual([(self.option(record, '--ablation'), self.option(record, '--seed')) for record in records], expected)
        for (mode, seed), record in zip(expected, records):
            self.assertEqual(record['cwd'], str(self.project))
            self.assertEqual(record['env']['PYTHONHASHSEED'], seed)
            self.assertEqual(self.option(record, '--batch-size'), '2')
            self.assertEqual(self.option(record, '--llm-type'), 'qwen')
            self.assertEqual(self.option(record, '--attn-implementation'), 'eager')
            self.assertEqual(self.resolved(record, self.option(record, '--output-dir')), checkpoint_root / mode / ('seed_' + seed))

    def test_evaluation_loads_each_variant_checkpoint_with_matching_output(self):
        checkpoint_root = self.directory / 'custom checkpoint root'
        output_root = self.directory / 'custom evaluation root'
        for mode in VARIANTS:
            checkpoint = checkpoint_root / mode / 'seed_42' / 'checkpoint.pt'
            checkpoint.parent.mkdir(parents = True)
            checkpoint.touch()

        result, records = self.launch(environment = {'MODE': 'eval', 'CHECKPOINT_ROOT': str(checkpoint_root),
                                                     'OUT_ROOT': str(output_root)})

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(records), len(VARIANTS))
        for mode, record in zip(VARIANTS, records):
            self.assertEqual(record['cwd'], str(self.project))
            self.assertEqual(record['argv'][:3], ['--standalone', '--nproc-per-node=gpu', 'eval.py'])
            self.assertEqual(record['env']['PYTHONHASHSEED'], '42')
            self.assertEqual(self.resolved(record, record['env']['IPMOE_CHECKPOINT']), checkpoint_root / mode / 'seed_42' / 'checkpoint.pt')
            self.assertEqual(self.option(record, '--judge'), 'exact_matching')
            self.assertEqual(self.option(record, '--data'), 'POPE')
            work_dir = self.resolved(record, self.option(record, '--work-dir'))
            self.assertEqual(work_dir, output_root / mode / 'seed_42')
            self.assertTrue((work_dir / 'eval.log').is_file())

    def test_preflight_rejects_invalid_lists_modes_paths_and_owned_arguments(self):
        help_result, help_records = self.launch(['--help'])
        self.assertEqual(help_result.returncode, 0, help_result.stdout + help_result.stderr)
        self.assertEqual(help_records, [])
        self.assertIn('ABLATIONS', help_result.stdout)
        cases = [([], {'MODE': 'invalid'}), ([], {'ABLATIONS': ''}), ([], {'ABLATIONS': ' \t '}),
                 ([], {'ABLATIONS': 'full unknown'}), ([], {'ABLATIONS': 'full full'}),
                 ([], {'SEEDS': ''}), ([], {'SEEDS': ' \t '}), ([], {'SEEDS': '-1'}),
                 ([], {'SEEDS': '2.5'}), ([], {'SEEDS': '42 invalid'}),
                 ([], {'CHECKPOINT_DIR': 'legacy checkpoints'}), ([], {'OUT': 'legacy results'})]
        for option, value in (('--ablation', 'full'), ('--seed', '42'), ('--output-dir', 'custom'), ('--work-dir', 'custom'),
                              ('--ablat', 'full'), ('--see', '42'), ('--out', 'custom'), ('--work', 'custom')):
            cases.extend([([option, value], {}), ([option + '=' + value], {})])
        for option in ('--model', '--mo'):
            cases.extend([([option, 'OtherModel'], {'MODE': 'eval'}), ([option + '=OtherModel'], {'MODE': 'eval'})])
        for arguments, environment in cases:
            with self.subTest(arguments = arguments, environment = environment):
                result, records = self.launch(arguments, environment)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(records, [])

    def test_failed_variant_does_not_prevent_remaining_variants_and_fails_wrapper(self):
        result, records = self.launch(environment = {'SMOKE_FAIL_VARIANT': 'spectral_only'})

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([self.option(record, '--ablation') for record in records], list(VARIANTS))
        self.assertIn('spectral_only', result.stdout + result.stderr)
        final_output = self.project / 'checkpoint' / 'ablations' / 'direct_image' / 'seed_42'
        self.assertTrue((final_output / 'train.log').is_file())

    def test_interrupt_status_stops_all_remaining_variants_and_seeds(self):
        for variant in VARIANTS:
            for seed in ('42', '43'):
                checkpoint = self.project / 'checkpoint' / 'ablations' / variant / ('seed_' + seed) / 'checkpoint.pt'
                checkpoint.parent.mkdir(parents = True, exist_ok = True)
                checkpoint.touch()
        for mode in ('train', 'eval'):
            for status in (130, 143):
                with self.subTest(mode = mode, status = status):
                    result, records = self.launch(environment = {'MODE': mode, 'SEEDS': '42 43',
                                                                 'SMOKE_EXIT_STATUS': str(status)})

                    self.assertEqual(result.returncode, status, result.stdout + result.stderr)
                    self.assertEqual(len(records), 1)
                    record = records[0]
                    self.assertEqual(record['env']['PYTHONHASHSEED'], '42')
                    if mode == 'train':
                        self.assertEqual(self.option(record, '--ablation'), 'full')
                    else:
                        checkpoint = self.resolved(record, record['env']['IPMOE_CHECKPOINT'])
                        self.assertEqual(checkpoint, self.project / 'checkpoint' / 'ablations' / 'full' / 'seed_42' / 'checkpoint.pt')


if __name__ == '__main__':
    unittest.main()
