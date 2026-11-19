"""Compare complete repair APIs from two independent verified checkouts."""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tracemalloc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout")
    parser.add_argument("sha")
    parser.add_argument("mode", choices=("semantic", "timing", "allocation"))
    args = parser.parse_args()
    checkout = Path(args.checkout).resolve()
    actual_sha = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True,
    ).strip()
    assert actual_sha == args.sha, (actual_sha, args.sha)
    root = checkout / "3MF/agent-harness"
    sys.path.insert(0, str(root))
    import numpy as np
    from cli_anything.threemf.core import repair
    from cli_anything.threemf.core.parser import MeshData
    assert Path(repair.__file__).resolve().is_relative_to(root)

    def array_digest(array):
        return {"dtype": str(array.dtype), "shape": array.shape,
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}

    def mesh_fixture(n, distinct, seed=1901, dtype=np.float64):
        rng = np.random.default_rng(seed)
        if not n:
            return np.empty((0, 3), dtype=dtype), np.empty((0, 3), dtype=np.int32)
        points = rng.uniform(-100, 100, (distinct, 3)).astype(dtype)
        vertices = points[np.arange(n) % distinct]
        rng.shuffle(vertices)
        triangles = rng.integers(0, n, (max(1, n // 2), 3), dtype=np.int32)
        return vertices, triangles

    provenance = {"source": actual_sha, "module": str(repair.__file__),
                  "python": sys.version, "numpy": np.__version__}
    if args.mode == "semantic":
        results = []
        for seed in range(4):
            for n in (0, 1, 25, 100):
                for dtype in (np.float32, np.float64):
                    for layout in ("C", "F", "strided"):
                        vertices, triangles = mesh_fixture(n, max(1, n // 2), seed, dtype)
                        if layout == "F":
                            vertices = np.asfortranarray(vertices)
                            triangles = np.asfortranarray(triangles)
                        elif layout == "strided":
                            storage = np.empty((n, 6), dtype=dtype)
                            storage[:, ::2] = vertices
                            vertices = storage[:, ::2]
                            triangles = triangles[:, ::-1]
                        before = [array_digest(vertices), array_digest(triangles)]
                        for decimals in (-1, 0, 6):
                            unique, remapped, removed = repair.merge_duplicate_vertices(
                                vertices, triangles, decimals=decimals,
                            )
                            results.append({"case": [seed, n, str(dtype), layout, decimals],
                                            "vertices": array_digest(unique),
                                            "triangles": array_digest(remapped), "removed": removed})
                            assert before == [array_digest(vertices), array_digest(triangles)]
                        attrs = tuple({"marker": str(i)} for i in range(len(triangles)))
                        mesh = MeshData("17", "preserved", vertices, triangles, attrs)
                        fixed, report = repair.repair_mesh(mesh)
                        results.append({"repair": [seed, n, str(dtype), layout],
                                        "vertices": array_digest(fixed.vertices),
                                        "triangles": array_digest(fixed.triangles),
                                        "attrs": fixed.triangle_attributes, "report": report,
                                        "id": fixed.object_id, "name": fixed.name})
                        assert before == [array_digest(vertices), array_digest(triangles)]
        for dtype in (np.float32, np.float64):
            vertices = np.array([[np.nan, 1, 0], [np.inf, 0, 1], [0, -0., 0],
                                 [np.nan, 1, 0], [-np.inf, 0, 1], [-0., 0, 0]], dtype=dtype)
            triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
            unique, remapped, removed = repair.merge_duplicate_vertices(vertices, triangles)
            results.append({"special": str(dtype), "vertices": array_digest(unique),
                            "triangles": array_digest(remapped), "removed": removed})
        print(json.dumps({"provenance": provenance, "results": results}, sort_keys=True))
        return

    configurations = [(1000, 1000), (100000, 100000), (500000, 500000),
                      (500000, 250000), (500000, 50000)]
    rows = []
    for n, distinct in configurations:
        vertices, triangles = mesh_fixture(n, distinct)
        mesh = MeshData("1", "timed", vertices, triangles)
        before = [array_digest(vertices), array_digest(triangles)]
        result, report = repair.repair_mesh(mesh)
        expected = [array_digest(result.vertices), array_digest(result.triangles), report]
        del result
        if args.mode == "allocation":
            gc.collect()
            tracemalloc.start()
            result, report = repair.repair_mesh(mesh)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append({"n": n, "distinct": distinct, "traced_current_B": current,
                         "traced_peak_B": peak, "output": expected})
            del result
        else:
            start = time.perf_counter()
            repair.repair_mesh(mesh)
            elapsed = time.perf_counter() - start
            iterations = max(1, min(500, math.ceil(0.1 / elapsed)))
            samples = []
            gc.disable()
            try:
                for _ in range(3):
                    start = time.perf_counter_ns()
                    for _ in range(iterations):
                        result, report = repair.repair_mesh(mesh)
                    samples.append((time.perf_counter_ns() - start) / iterations)
            finally:
                gc.enable()
            assert [array_digest(result.vertices), array_digest(result.triangles), report] == expected
            rows.append({"n": n, "distinct": distinct, "iterations": iterations,
                         "samples_ns": samples, "median_ns": statistics.median(samples),
                         "output": expected})
            del result
        assert before == [array_digest(vertices), array_digest(triangles)]
    print(json.dumps({"provenance": provenance, "rows": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
