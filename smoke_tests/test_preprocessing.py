"""Exercise parquet streaming and source validation without downloading data."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

import preprocessing


class PreprocessingSmokeTests(unittest.TestCase):
    def test_streams_requested_columns_and_rejects_incomplete_sources(self):
        source = dict(repo = 'synthetic', revision = 'local', patterns = ('part-*.parquet',),
                      expected_shards = 2, expected_rows = 3)
        with tempfile.TemporaryDirectory() as directory:
            for index, questions in enumerate((['first', 'second'], ['third'])):
                table = pa.table({'question': questions, 'image': [b'undecoded'] * len(questions)})
                pq.write_table(table, Path(directory) / f'part-{index}.parquet')

            with patch.dict(preprocessing.SOURCES, synthetic = source), \
                 patch.object(preprocessing, 'snapshot_download', return_value = directory):
                rows = list(preprocessing.iter_rows('synthetic', ['question'], desc = 'synthetic', batch_size = 1))
                self.assertEqual(rows, [{'question': value} for value in ('first', 'second', 'third')])

                source['expected_rows'] = 4
                with self.assertRaisesRegex(RuntimeError, 'Expected 4 rows'):
                    list(preprocessing.iter_rows('synthetic', ['question']))

                source['expected_shards'] = 3
                with self.assertRaisesRegex(RuntimeError, 'Expected 3 parquet shards'):
                    list(preprocessing.iter_rows('synthetic', ['question']))


if __name__ == '__main__':
    unittest.main()
