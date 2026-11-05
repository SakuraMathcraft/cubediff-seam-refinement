# Research notes

## What was held fixed

The same frozen seven-channel CubeDiff checkpoint, condition and prompt, seed, six-face attention, joint VAE decode, 512-pixel faces, DDIM schedule and CFG scale 3.5 were used in paired experiments. Each run called the scheduler 50 times and the UNet 100 times. The graph update acts on five target-face epsilon predictions during the final 15 steps, using 12 projected solver iterations. The pixelwise radius is expressed as a fraction of the per-face latent standard deviation.

The 0.25 primary experiment used seeds 6, 7 and 8. The 0.35 ablation used already inspected seed 8 and reran both arms. All four scenes were retained in both experiments. The primary implementation is the default because it has three-seed evidence, not because it won a retrospective search.

## Metric and evidence

For each of 12 undirected cube edges, we sampled 128 positions along the edge and 17 offsets within the common field of view. The exact same rays were bilinearly sampled in both faces. Absolute RGB differences were averaged across samples and channels, then equally across edges. RGB values were in [0, 1]. The reported relative gain is `1 - mean(refined L1) / mean(baseline L1)`.

Quality decisions came from a condition-image check (original crop or fixed codec anchor, with LPIPS and coarse RGB thresholds) and a baseline-relative LPIPS envelope applied to each target face. The envelope was calibrated from small brightness changes and border blur. It is a limited content-retention diagnostic and has rejected some mild transformations in earlier controls. It does not certify that an output scene is realistic.

[results.json](results.json) contains every pair's raw edge metrics, quality decision and evaluated PNG fingerprints, plus summary statistics and local experiment fingerprints. The underlying full experiment archive is kept locally. The standalone CLI regenerates faces and measures raw overlap; LPIPS scoring resources and full case logs are not included. The timing measurements in the JSON were taken with no reference observer, include setup, generation, output writes and cleanup, and exclude shared loading and post-run paired LPIPS. They are not portable quotas.

The exploratory paired absolute-gain interval was [0.01083, 0.02884], calculated on three seed-level means. Four scenes do not supply four extra independent seeds. Separately, baseline standard deviation was 0.01774, so the gain-to-baseline-variation ratio (1.1184) failed the 3× diagnostic. No cases or edges were discarded to improve that outcome.

## Provenance of figures and package

- [Room comparison](../assets/room-comparison.webp): primary experiment, dev_002, seed 6, both arms. The associated condition is a crop of Poly Haven's Lebombo HDRI. Display panels are resized equirectangular outputs, not scored inputs.
- [City limitation](../assets/city-limit.webp): primary experiment, dev_003, seed 8, refined arm. The roof/sky boundary remains discontinuous.
- Baseline implementation lineage: [OpenCubeDiff revision `1a3ac656`](https://github.com/Juan5713/OpenCubeDiff/tree/1a3ac656ed974120899d792699a9f310f10ad888). Its [MIT notice](../licenses/OpenCubeDiff-MIT.txt) covers upstream-derived code. The separate checkpoint is not redistributed.

The published package is a self-contained extraction of the inference and metric code. The study used Windows, Python 3.12, CUDA 12.8 and Diffusers 0.34.0. The 52 checks reported in the full local project are distinct from the small checkpoint-free suite shipped here. Clean Linux, whole-pipeline bitwise replay, and production deployment were not verified.
