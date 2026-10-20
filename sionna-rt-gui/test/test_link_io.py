"""Result I/O stays off the render thread and old jobs cannot leak into the UI."""
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace as N
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sionna_rt_gui.link_jobs import LinkJobs


class LinkIOTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2] / '.verification' / 'link-io-tests'
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.folder = Path(self.temp.name)
        self.assertEqual(self.folder.resolve().parent, root.resolve())
        self.settings = dict(scene='a.xml', panel=0,
            rx=[('rx-0', [1, 2, 3], [0, 0, 0])])
        self.app = N(cfg=N(config_path=str(self.folder / 'config.yaml')),
            demo_page=1, scene=N(transmitters={'tx': 1}, receivers={'rx': 1}),
            demo_link_settings=None, animation_config=N(playing=False),
            beam_demo=N(signature=None), update_demo_cir=Mock())
        self.jobs = LinkJobs(self.app)
        self.jobs.draw_segments = Mock()
        self.gates = []

    def tearDown(self):
        for gate in self.gates:
            gate.set()
        self.jobs.close()
        self.jobs._io.shutdown(wait=True)
        self.temp.cleanup()

    def ready(self, name='result'):
        output = self.folder / name
        np.savez(output.with_suffix('.npz'), amplitudes=np.array([1.5]),
            delays=np.array([2.5]), starts=np.zeros((1, 3)), ends=np.ones((1, 3)))
        output.with_suffix('.json').write_text(json.dumps({'seconds': .01}))
        job = dict(output=str(output), settings=self.settings, submitted=time.monotonic())
        self.jobs.pending = job
        self.jobs.desired = self.settings
        self.jobs._result_future = self.jobs._io.submit(self.jobs.read_result, job)
        self.jobs._result_future.result(timeout=3)
        return output

    def tick(self, settings):
        with patch('sionna_rt_gui.link_jobs.link_signature', return_value=settings), \
             patch('sionna_rt_gui.link_jobs.ps.has_curve_network', return_value=False), \
             patch('sionna_rt_gui.link_jobs.ui.IsMouseDown', return_value=True):
            self.jobs.tick(self.app)

    def test_completed_result_is_loaded_cleaned_and_applied_on_caller_thread(self):
        output = self.ready()
        self.assertFalse(output.with_suffix('.npz').exists())
        self.assertFalse(output.with_suffix('.json').exists())
        draw_threads = []
        self.jobs.draw_segments.side_effect = lambda *args: draw_threads.append(threading.get_ident())
        self.tick(self.settings)
        np.testing.assert_array_equal(self.app.demo_link_cir[0], [1.5])
        self.assertEqual(self.app.demo_link_settings, self.settings)
        self.assertEqual(draw_threads, [threading.get_ident()])
        self.app.update_demo_cir.assert_called_once()

    def test_panel_changed_while_reading_rejects_old_result(self):
        self.ready()
        self.tick({**self.settings, 'panel': 1})
        self.assertIsNone(self.app.demo_link_settings)
        self.jobs.draw_segments.assert_not_called()
        self.app.update_demo_cir.assert_not_called()

    def test_close_does_not_wait_for_blocked_io_and_cleanup_runs_later(self):
        output = self.ready()
        # Leave another result on disk for close's deferred cleanup.
        output.with_suffix('.json').write_text('{}')
        gate, started = threading.Event(), threading.Event()
        self.gates.append(gate)

        def blocking_io():
            started.set()
            gate.wait(2)

        self.jobs._result_future = self.jobs._io.submit(blocking_io)
        self.assertTrue(started.wait(1))
        before = time.perf_counter()
        self.jobs.close()
        self.assertLess(time.perf_counter() - before, .25)
        self.assertIsNone(self.jobs.pending)
        self.assertIsNone(self.jobs._result_future)
        self.assertTrue(output.with_suffix('.json').exists())
        gate.set()
        self.jobs._io.submit(lambda: None).result(timeout=3)
        self.assertFalse(output.with_suffix('.json').exists())
