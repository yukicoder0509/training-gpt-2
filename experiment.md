# Experiments

W&B project: `cerulean-labs/gpt2-training` on `https://app.forge.coreweave.com`.

## 2026-10-04 — Peak learning rate sweep (1×, 5×, 10×)

**Question:** Is the GPT-1 peak LR (2.5e-4) too low for a ~630-step, 30-min run?

### Setup

Common to all runs (`train.py` @ uncommitted changes on top of `e0cd366`):

| Setting | Value |
|---|---|
| Model | GPT-2 124M, from config (random init) |
| Data | Unfiltered C4 `en`, 20 shards (~3B tokens), `/work/$USER/c4_gpt2` |
| Context | 1024 tokens |
| Global batch | 512 sequences (32/device × 8 grad accum × 2 GPUs) ≈ 524K tokens/step |
| Optimizer | AdamW, β=(0.9, 0.95), ε=1e-8, weight decay 0.1 |
| Schedule | Linear warmup, then cosine to 10% of peak (`cosine_with_min_lr`, `min_lr_rate=0.1`) |
| Precision | bf16, 2 GPUs (`dev` partition, `run.sbatch`) |
| Eval | 5,000 C4 validation docs, every 20 steps |

Per-run:

| Run | Job | W&B run | Peak LR | Min LR | Steps | Warmup |
|---|---|---|---|---|---|---|
| Old baseline | 494928 | `nd38pu46` | 2.5e-4 | 2.5e-5 | 789 | 100 (13%) |
| 1× control | 495161 | `f5bcbd7t` | 2.5e-4 | 2.5e-5 | 631 | 63 (10%) |
| 5× | 495164 | `ax1zqw7w` | 1.25e-3 | 1.25e-4 | 631 | 63 (10%) |
| 10× | 495165 | `uhzs1is5` | 2.5e-3 | 2.5e-4 | 631 | 63 (10%) |

Steps differ because `TIME_BUDGET` was cut from 1500 s to 1200 s (the old baseline timed
out during the Hub upload). The 1× control isolates the effect of LR at the new step count.

Command:
```
sbatch --job-name=gpt2-c4-lr$LR run.sbatch --learning_rate=$LR --run_name=lr$LR-wu10pct
```

### Results

| Run | Final eval loss | Final eval ppl | Final train loss | Max grad norm |
|---|---|---|---|---|
| Old baseline (789 steps) | 4.996 | 147.8 | 5.090 | 2.57 |
| 1× control | 5.222 | 185.3 | 5.296 | 2.10 |
| **5×** | **4.489** | **89.0** | **4.590** | 1.87 |
| 10× | 5.091 | 162.6 | 5.157 | 1.57 |

Eval loss at the same step:

| Step | Old baseline | 1× | 5× | 10× |
|---|---|---|---|---|
| 20 | 9.276 | 8.961 | 7.659 | **7.592** |
| 60 | 7.227 | 7.047 | **6.732** | 6.938 |
| 100 | 6.579 | 6.473 | **6.376** | 6.560 |
| 200 | 6.039 | 6.009 | **5.801** | 6.141 |
| 300 | 5.711 | 5.703 | **5.315** | 5.802 |
| 400 | 5.448 | 5.463 | **4.940** | 5.514 |
| 500 | 5.247 | 5.318 | **4.683** | 5.275 |
| 600 | 5.115 | 5.239 | **4.519** | 5.123 |
| 620 | 5.097 | 5.228 | **4.502** | 5.102 |
| 780 | 5.000 | — | — | — |

### Findings

- **5× (1.25e-3) is best by a wide margin:** eval loss 4.489 vs 4.996 for the old baseline,
  despite 158 fewer steps. It leads from step ~60 onward, and its loss falls smoothly
  with no spikes.
- **2.5e-4 is far too low for this budget.** The 1× control tracks the old baseline closely at
  matched steps; it ends worse only because it is cut off earlier. Moving warmup from 100 to
  63 steps had little visible effect.
- **10× (2.5e-3) is past the optimum but stable:** it leads very early (step 20), then
  falls behind 5×, with no loss spikes or divergence, and ends about level with the old
  baseline. Its LR floor (2.5e-4) equals the 1× peak, so it is still taking large steps
  at the end.
- The optimum is likely between 5× and 10×, probably nearer 5×.

### Caveats

- One seed per LR.
- ~0.12 epoch of unfiltered data.
- The old baseline differs from the 1× control in both step count and warmup.

### Operational notes

- Three 2-GPU jobs landed on the same node and collided on `accelerate`'s default port
  29500 (`EADDRINUSE`). Fixed in `run.sbatch` with `--main_process_port=$((29500 + SLURM_JOB_ID % 1000))`;
  the 5× / 10× runs were resubmitted (495162/495163 → 495164/495165).
- Train time ≈ 1364 s, but three concurrent 498 MB Hub uploads were slow: 5× finished at
  1789 s, and 10× hit the 30-min limit (1810 s, `TIMEOUT`) right after its upload committed.
- All runs push to the same repo, `cerulean-works/ben-lab4-c4-gpt2`, so the 10× model
  (`bbc4001f21`) ended up on top. The 5× weights (`2007e4955f`) were restored as the
  latest version in commit `0abbf2a792` (a server-side copy, no retraining).

### Next

- Narrow the LR: try 7.5e-4 and 1.75e-3.
- Re-run the best LR on the DCLM-filtered data (`~/c4_gpt2_dclm`; prep job 495041).

## 2026-10-04 — DCLM fastText quality filtering

**Question:** Does training on the top ~10% of C4 by the DCLM quality classifier beat the
unfiltered C4 data at the best LR (1.25e-3)?

### Data prep (job 495041, `prepare.sbatch`, 12 CPUs, 2h10m)

| | |
|---|---|
| Classifier | `mlfoundations/fasttext-oh-eli5` (`openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin`), score = P(`__label__hq`), newlines replaced with spaces |
| Scored | C4 `en` train shards 0–199: 71,263,455 docs / 34.1B tokens |
| Threshold | hq ≥ 0.0931 (chosen so the kept docs total 3B tokens) |
| Kept | 6,702,562 docs (9.4%) / 3,000,000,973 tokens |
| Output | `~/c4_gpt2_dclm/train.bin` (6.0 GB); `val.bin` = the same unfiltered 5k-doc C4 validation set as before |
| Score cache | `~/c4_gpt2_dclm/dclm_scores/*.npz` (817 MB, per-doc score + token count); re-cutting at a new threshold skips scoring (~10–15 min) |

Command:
```
sbatch prepare.sbatch          # --num_train_shards 200 --target_train_tokens 3000000000
```

### Training run

Same as the 5× run (495164) except `--data_dir`:

| Run | Job | W&B run | Data | Final eval loss | Final eval ppl | Final train loss |
|---|---|---|---|---|---|---|
| 5×, unfiltered C4 | 495164 | `ax1zqw7w` | `/work/$USER/c4_gpt2` | **4.489** | **89.0** | 4.590 |
| 5×, DCLM top 9.4% | 495883 | `pjpo7ugf` | `~/c4_gpt2_dclm` | 4.590 | 98.5 | **4.482** |

Command:
```
sbatch --job-name=gpt2-c4-dclm-lr1.25e-3 run.sbatch --learning_rate=1.25e-3 \
    --run_name=dclm-lr1.25e-3-wu10pct --data_dir=$HOME/c4_gpt2_dclm
```

Eval loss on the unfiltered C4 validation set at the same step:

| Step | Unfiltered | DCLM | Δ |
|---|---|---|---|
| 20 | 7.659 | 7.744 | +0.085 |
| 100 | 6.376 | 6.442 | +0.066 |
| 200 | 5.801 | 5.885 | +0.084 |
| 300 | 5.315 | 5.427 | +0.112 |
| 400 | 4.940 | 5.049 | +0.109 |
| 500 | 4.683 | 4.775 | +0.092 |
| 620 | 4.502 | 4.604 | +0.102 |

### Findings

- **On C4 validation loss, DCLM filtering hurts by about 0.1 nats** (perplexity 98.5 vs 89.0).
  The gap is roughly constant from the start, so it looks like a data distribution
  mismatch rather than slower learning.
- The DCLM run's *train* loss is lower (4.48 vs 4.59): the filtered text is easier to
  predict, but it is a different distribution from the unfiltered C4 the eval set is drawn from.
- The DCLM paper measured gains on downstream tasks (e.g. MMLU, HellaSwag), not on raw C4
  perplexity. If the target metric is C4 validation loss/ppl, unfiltered C4 is the better
  training data. This run does not test whether downstream scores improve.

### Operational notes

- Train time 1370 s; the 498 MB upload averaged ~480 kB/s and the job hit the 30-min limit
  (`TIMEOUT`, 1817 s) at the moment the Hub commit landed (`dd22a4a2f8`). That commit
  replaced the restored 5× model (`0abbf2a792`) on the Hub.
- `/work` quota is full (100 G), so the DCLM data and the 2.3 GB classifier live on `/home`.
- Login-node test of `prepare_data.py` hung (likely `hf_hub_download` inside forked
  workers + unbatched tokenization ~6 GB/worker). Fixed by resolving shard paths in the
  parent and tokenizing in 1k-doc batches; heavy work runs only via `sbatch`.

### Next

- If C4 val ppl is the metric: drop DCLM filtering, or try a softer cut at the same 3B
  tokens by scoring fewer shards (e.g. `--num_train_shards 60` ≈ top 30%, `40` ≈ top 45%;
  scores are cached, so only the ~10–15 min tokenize pass runs), or mix filtered + unfiltered data.
- Shorten upload time or the training budget: single-run uploads still take ~6–7 min at
  current Hub speeds, leaving no margin under the 30-min limit.

## 2026-10-04 — Constant LR after warmup vs cosine decay

**Question:** The model is far from convergence after ~630 steps, so does keeping the LR at
its peak (no decay) beat cosine decay to 10%?

### Setup

Same as the 5× run (495164: unfiltered C4, 631 steps, 63 warmup, peak 1.25e-3) except the
schedule: `lr_scheduler_type="constant_with_warmup"` (`get_constant_schedule_with_warmup`),
i.e. linear warmup to 1.25e-3, then constant to the end. `--learning_rate` now defaults to 1.25e-3.

```
sbatch --job-name=gpt2-c4-const-lr1.25e-3 run.sbatch --learning_rate=1.25e-3 --run_name=const-lr1.25e-3-wu10pct
```

### Results

| Run | Job | W&B run | Schedule | Final LR | Final eval loss | Final eval ppl | Final train loss |
|---|---|---|---|---|---|---|---|
| Cosine 5× | 495164 | `ax1zqw7w` | cosine → 10% | 1.25e-4 | 4.489 | 89.0 | 4.590 |
| **Constant 5×** | 496010 | `yr73yhyq` | constant | 1.25e-3 | **4.287** | **72.7** | **4.385** |

Eval loss at the same step:

| Step | Cosine | Constant | Δ |
|---|---|---|---|
| 20 | 7.659 | 7.656 | −0.003 |
| 100 | 6.376 | 6.420 | +0.044 |
| 200 | 5.801 | 5.818 | +0.017 |
| 300 | 5.315 | 5.366 | +0.051 |
| 400 | 4.940 | 4.962 | +0.022 |
| 500 | 4.683 | 4.585 | −0.098 |
| 560 | 4.569 | 4.406 | −0.163 |
| 600 | 4.519 | 4.332 | −0.187 |
| 620 | 4.502 | 4.307 | −0.195 |

### Findings

- **Constant LR wins by 0.20 nats** (ppl 72.7 vs 89.0); this is the best run so far.
- Through step ~400 the runs are within ~0.05 of each other, with cosine slightly ahead.
  After that, cosine's LR falls toward 1.25e-4 and its progress slows, while the constant
  run keeps dropping fast.
- No loss spikes; max grad norm 1.96 (vs 1.87). The constant run is still improving
  steeply at the end, so the budget, not the LR, is the binding constraint. This supports
  trying an even higher peak LR.
- This contradicts the usual result that LR decay helps the final loss; in this
  short, heavily undertrained regime, the decay costs more progress than it gives back.

### Caveats

- One seed.
- Without a final decay the end-of-run loss is noisier; a short cooldown could still add
  gains on top (see Next).

### Operational notes

- Train time ~1370 s, script finished at 1785 s; the Hub commit (`bbaaa0e3f5`) landed
  ~15 s before the 30-min limit. The constant-LR model is now the latest Hub version.

### Next

- Higher constant LR: 2e-3 and 2.5e-3. The 10× cosine run never spiked, it only trailed 5×,
  so it is worth retesting under a constant schedule.
- Warmup-stable-decay (WSD): constant at peak, then a short linear decay over the last
  ~10–20% of steps. This often beats both constant and cosine for a fixed budget.

## 2026-10-04 — Higher constant LR, and warmup-stable-decay (WSD)

**Questions:** (1) With a constant schedule, does a higher peak than 1.25e-3 help?
(2) Does a short final decay on top of the constant schedule help?

### Setup

All on unfiltered C4, 631 steps, 63 warmup (10%), otherwise as before. New flag
`--decay_frac`: 0 = constant after warmup; > 0 = `warmup_stable_decay`
(`get_wsd_schedule`), holding the peak then decaying linearly to 0 over that fraction of
the final steps. For 0.2 that is a hold from step 63 to 505, then decay over steps 505–631 (126 steps).

```
sbatch --job-name=gpt2-c4-const-lr2e-3   run.sbatch --learning_rate=2e-3   --run_name=const-lr2e-3-wu10pct
sbatch --job-name=gpt2-c4-const-lr2.5e-3 run.sbatch --learning_rate=2.5e-3 --run_name=const-lr2.5e-3-wu10pct
sbatch --job-name=gpt2-c4-wsd-lr1.25e-3  run.sbatch --learning_rate=1.25e-3 --decay_frac=0.2 --run_name=wsd20-lr1.25e-3-wu10pct
```

### Results

| Run | Job | W&B run | Schedule | Peak LR | Final eval loss | Final eval ppl | Final train loss |
|---|---|---|---|---|---|---|---|
| Constant (previous best) | 496010 | `yr73yhyq` | constant | 1.25e-3 | 4.287 | 72.7 | 4.385 |
| Constant | 496211 | `3ouu1o1l` | constant | 2e-3 | 4.510 | 90.9 | 4.594 |
| Constant | 496209 | `u4c0jgqy` | constant | 2.5e-3 | 4.828 | 125.0 | 4.933 |
| **WSD 20%** | 496210 | `nmo4kpdn` | warmup–stable–decay | 1.25e-3 | **4.219** | **68.0** | **4.307** |

Eval loss at the same step:

| Step | Const 1.25e-3 | Const 2e-3 | Const 2.5e-3 | WSD 1.25e-3 |
|---|---|---|---|---|
| 20 | 7.656 | 7.522 | **7.521** | 7.642 |
| 100 | 6.420 | 6.392 | 6.555 | **6.366** |
| 200 | 5.818 | 5.958 | 6.195 | **5.784** |
| 300 | 5.366 | 5.593 | 5.835 | **5.308** |
| 400 | 4.962 | 5.261 | 5.548 | **4.902** |
| 500 | 4.585 | 4.880 | 5.235 | **4.526** |
| 560 | 4.406 | 4.653 | 5.033 | **4.329** |
| 600 | 4.332 | 4.556 | 4.915 | **4.252** |
| 620 | 4.307 | 4.497 | 4.860 | **4.226** |

### Findings

- **WSD (decay over the last 20%) is the new best: 4.219 / ppl 68.0**, 0.068 better than
  constant 1.25e-3.
- WSD was already ~0.06 ahead *before* its decay began (step 500: 4.526 vs 4.585), at an
  identical LR. So this is roughly half run-to-run noise and half decay gain; the gain from the
  decay itself is about 0.02–0.03. Only one seed each, so the noise level is not pinned down.
- **Higher constant LR hurts:** 2e-3 → 4.510, 2.5e-3 → 4.828. They lead only in the first
  ~20–100 steps. The gap then widens steadily, with no spikes or divergence; the runs just
  make slower progress. With a constant schedule the best peak is ≤ 1.25e-3, so the
  earlier "not close to a minimum, so be aggressive" idea holds for *keeping* the LR high, not
  for raising the peak.
- Across schedules, 1.25e-3 is the best peak tested so far: it won under cosine (vs 2.5e-4 and 2.5e-3)
  and under constant (vs 2e-3 and 2.5e-3).

### Operational notes

- First 2e-3 attempt (496208) died on **25a-hgpn001**: both ranks were assigned the same GPU
  (NCCL "Multiple Ranks are using the same GPU/Partition"). hgpn001 is now in
  `run.sbatch --exclude`; resubmitted as 496211.
- All three runs on different nodes finished in time (26–28.5 min). Hub commits: WSD
  `febe8607c2` (19:35:27 local); 2e-3 and 2.5e-3 committed one second apart at 19:37:36–37
  (`667c11b12f`, `f604298e05`), so the Hub's latest version was one of the worse high-LR models.
  The WSD weights were restored as the latest version in commit `d95c12cccf` (a server-side copy).

### Next

- Peak LR between 7.5e-4 and 1.25e-3 under WSD (e.g. 8e-4, 1e-3).
- Longer decay under WSD (30–40%) at 1.25e-3.
- A second seed for constant vs WSD to measure run-to-run noise (~0.05?).

## 2026-10-04 — WSD grid: peak LR × decay length

**Question:** Under warmup-stable-decay, what is the best peak LR, and is a 30% decay better than 20%?

### Setup

Unfiltered C4, 631 steps, 63 warmup (10%), linear decay to 0, everything else as before.
Decay starts at step 505 (20%) or 442 (30%). Cell (1.25e-3, 20%) reuses run 496210.

```
for cfg in "7.5e-4 0.2" "7.5e-4 0.3" "1e-3 0.2" "1e-3 0.3" "1.25e-3 0.3" "1.5e-3 0.2" "1.5e-3 0.3"; do
  set -- $cfg; d=$(python3 -c "print(int(float('$2')*100))")
  sbatch --job-name=gpt2-c4-wsd$d-lr$1 run.sbatch --learning_rate=$1 --decay_frac=$2 --run_name=wsd$d-lr$1
done
```

### Results

Final eval loss (ppl):

| Peak LR | Decay 20% | Decay 30% |
|---|---|---|
| 7.5e-4 | 4.362 (78.4) | 4.404 (81.8) |
| 1e-3 | 4.266 (71.2) | 4.331 (76.0) |
| **1.25e-3** | **4.219 (68.0)** | 4.275 (71.9) |
| 1.5e-3 | 4.457 (86.2) | 4.301 (73.8) |

| Peak LR | Decay | Job | W&B run | Final eval | Final train | Max grad norm | Eval @420 | Eval @500 |
|---|---|---|---|---|---|---|---|---|
| 7.5e-4 | 20% | 496865 | `rttu80ib` | 4.362 | 4.455 | 1.96 | 4.934 | 4.747 |
| 7.5e-4 | 30% | 496866 | `2rinzqax` | 4.404 | 4.497 | 1.99 | 4.991 | 4.681 |
| 1e-3 | 20% | 496867 | `kvuasl3u` | 4.266 | 4.354 | 1.76 | 4.883 | 4.580 |
| 1e-3 | 30% | 496868 | `x9r6jn7l` | 4.331 | 4.421 | 2.38 | 4.916 | 4.592 |
| 1.25e-3 | 20% | 496210 | `nmo4kpdn` | **4.219** | **4.307** | 1.93 | 4.838 | 4.526 |
| 1.25e-3 | 30% | 496869 | `zx2oaead` | 4.275 | 4.363 | 1.81 | 4.854 | 4.517 |
| 1.5e-3 | 20% | 496870 | `ru7ydldy` | 4.457 | 4.549 | 2.91 | 5.131 | 4.886 |
| 1.5e-3 | 30% | 496871 | `06oym5he` | 4.301 | 4.388 | 2.00 | 4.966 | 4.580 |

### Findings

- **Best is still 1.25e-3 with 20% decay (4.219).** The LR optimum is around 1e-3 to 1.25e-3;
  7.5e-4 is too low in both columns.
- **20% decay beats 30% at every LR except 1.5e-3** (by 0.04–0.07). A longer decay gives up
  more steps at the peak LR than it gains.
- **Noise estimate:** at step 420 neither decay has started, so the two runs at each LR
  have identical schedules. Their gaps there are pure run-to-run noise: 0.057, 0.033, 0.016, and
  **0.165** for 7.5e-4, 1e-3, 1.25e-3 and 1.5e-3. Differences of ≲0.05 in this grid are not
  meaningful.
- **1.5e-3 is on the edge of stability:** its 20% run had the highest grad norm (2.91) and fell
  0.165 behind its identically scheduled twin before any decay. That one bad run is why 1.5e-3/20%
  looks worse than 1.5e-3/30%; it is not evidence that longer decay helps at high LR.

### Operational notes

- **Attempt 1 (496710–496729):** NCCL's bootstrap picked an unroutable interface on several
  nodes ("No route to host"), hung ~10 min, then crashed. Fixed with
  `export NCCL_SOCKET_IFNAME=lo` in `run.sbatch` (single-node jobs only).
- **Attempt 2 (496807–496814):** `/home` hit its 100 G quota. `results/` held 48 G of
  checkpoints (3 × 1.4 GB per run), and all 7 jobs died at the same second while saving step 100.
  Deleted `results/*` and set `save_strategy="no"` in `train.py`; the final model is still pushed to the Hub.
- **Attempt 3 (496865–496871):** all completed in 23.5–28 min, faster than before because no checkpoints are written.
- The 7 uploads replaced the best model on the Hub; it was restored as the latest version
  in `ace779669b` (copied from `d95c12cccf`, same weights as 496210).

### Next

- Seeds: rerun 1e-3/20% and 1.25e-3/20% with two more seeds each to separate them from noise.
- Shorter decay (10%) at 1.25e-3, since 20% beat 30%.

## 2026-10-04 — Shorter WSD decay (10%, 5%) at 1.25e-3

**Question:** 20% decay beat 30%. Is an even shorter decay better?

### Setup

Same as 496210 (unfiltered C4, 631 steps, 63 warmup, peak 1.25e-3), with `--decay_frac` 0.1
(63 decay steps, from step 568) and 0.05 (31 decay steps, from step 600).

```
sbatch --job-name=gpt2-c4-wsd10-lr1.25e-3 run.sbatch --learning_rate=1.25e-3 --decay_frac=0.1  --run_name=wsd10-lr1.25e-3
sbatch --job-name=gpt2-c4-wsd5-lr1.25e-3  run.sbatch --learning_rate=1.25e-3 --decay_frac=0.05 --run_name=wsd5-lr1.25e-3
```

### Results (all at peak 1.25e-3)

| Decay | Job | W&B run | Final eval loss | Final eval ppl | Final train loss | Max grad norm |
|---|---|---|---|---|---|---|
| 0% (constant) | 496010 | `yr73yhyq` | 4.287 | 72.7 | 4.385 | 1.96 |
| **5%** | 496949 | `j614qcw4` | **4.208** | **67.2** | **4.301** | 3.14 |
| 10% | 496947 | `7yinhs2d` | 4.230 | 68.7 | 4.315 | 1.90 |
| 20% | 496210 | `nmo4kpdn` | 4.219 | 68.0 | 4.307 | 1.93 |
| 30% | 496869 | `zx2oaead` | 4.275 | 71.9 | 4.363 | 1.81 |

Eval loss at the same step:

| Step | 5% | 10% | 20% | 30% | 0% |
|---|---|---|---|---|---|
| 420 | 4.821 | 4.848 | 4.838 | 4.854 | 4.887 |
| 500 | 4.508 | 4.559 | 4.526 | 4.517 | 4.585 |
| 560 | 4.360 | 4.395 | 4.329 | 4.362 | 4.406 |
| 600 | 4.301 | 4.283 | 4.252 | 4.299 | 4.332 |
| 620 | 4.230 | 4.241 | 4.226 | 4.279 | 4.307 |

### Findings

- **5%, 10% and 20% decay are a tie: 4.208 / 4.230 / 4.219, a spread of 0.022.**
  The 5% run is nominally best, but the margin is far inside the noise.
- **Noise check:** at step 500, the 5%, 10%, 20% and constant runs all have identical schedules
  (no decay has started), yet they span 4.508–4.585 (0.077). Run-to-run noise is about
  0.05–0.08, larger than every gap among the 5–20% decays.
- The decay is consistently worth something: every 5–20% run ends 0.06–0.08 below the
  constant run. 30% is slightly worse, losing too many peak-LR steps.
- The 5% run had a grad-norm spike to 3.14 (others ≤ 1.96) but no loss spike.
- **Practical choice:** a 10–20% decay at 1.25e-3. A shorter decay doesn't help measurably, and
  10–20% leaves more room if the step count changes.

### Operational notes

- Both runs completed in 25–27 min. The 5% run uploaded last (`c76275e8a1`), so the Hub's latest
  version is the 5% model, which is also the nominal best; no restore was needed.

### Next

- The schedule is now well tuned; further LR/decay tweaks are below the noise floor. Bigger levers:
  data (filtered/unfiltered mix), batch size vs step count, or throughput (more steps in 30 min).
- Multiple seeds would be needed to rank configs that differ by < 0.05.

## 2026-10-04/05 — Separate Hub push (746 steps) and weight decay 0.01 vs 0.1

**Changes:**
- `train.py` saves the final model to `~/gpt2_models/<run_name>` (`trainer.save_model`) instead of
  pushing it; `push_model.py` uploads it afterwards from the login node. The upload no longer counts
  against the 30-min job limit.
- The step budget now uses the measured 2.13 s/step (incl. eval every 20 steps) and
  `TIME_BUDGET = 1800 - 60 (startup) - 30 (save + W&B finish) - 120 (margin) = 1590 s` →
  **746 steps** (was 631). Warmup 74, 20% decay = 149 steps.
- `DeadlineCallback` stops training 90 s before the limit if a run is slow, so the model is still saved
  (ranks agree via all_reduce).
- New flags: `--weight_decay` (default now 0.01), `--save_dir`, `--optimizer` (`adam_mini` | `adamw`).

### Setup

AdamW, unfiltered C4, peak 1.25e-3, WSD 20% decay, 746 steps; only the weight decay differs.

```
sbatch --job-name=gpt2-c4-wd$WD run.sbatch --learning_rate=1.25e-3 --decay_frac=0.2 --weight_decay=$WD --run_name=wsd20-lr1.25e-3-wd$WD-746
```

### Results

| Run | Job | W&B run | Weight decay | Final eval loss | Final eval ppl | Final train loss | Max grad norm |
|---|---|---|---|---|---|---|---|
| **wd 0.01** | 497297 | `vdvpngw8` | 0.01 | **4.062** | **58.1** | 4.159 | 1.86 |
| wd 0.1 | 497298 | `03ixq3az` | 0.1 | 4.071 | 58.6 | 4.164 | 3.60 |

Eval loss at the same step:

| Step | wd 0.01 | wd 0.1 | Δ |
|---|---|---|---|
| 100 | 6.370 | 6.396 | −0.026 |
| 300 | 5.279 | 5.284 | −0.005 |
| 500 | 4.506 | 4.521 | −0.015 |
| 600 | 4.278 | 4.297 | −0.019 |
| 740 | 4.064 | 4.072 | −0.008 |

### Findings

- **The extra steps are the big win:** 746 steps at the best schedule gives eval loss 4.06–4.07 vs
  4.219 for the same schedule at 631 steps (−0.15; ppl 58 vs 68).
- **Weight decay 0.01 vs 0.1 is a tie** (Δ 0.009, inside the ~0.05–0.08 noise). wd 0.01 is ahead
  at every checkpoint by 0.005–0.026, so it may be slightly better, but this isn't conclusive. At
  < 0.15 epoch the model isn't overfitting, so there's little for weight decay to regularize.
- Timing: both jobs finished in 1653–1656 s total (saved at ~1616 s), ~2.5 min under the limit;
  the deadline safety stop was not triggered.

### Hub

- Pushed **wd 0.01 (497297)** with `push_model.py` (44 s from the login node): commit `7945c6fc31`,
  sha256 verified against the local file. Kept locally at `~/gpt2_models/wsd20-lr1.25e-3-wd0.01-746`.
- Deleted the wd 0.1 model (`~/gpt2_models/wsd20-lr1.25e-3-wd0.1-746`) and the empty `results/` dirs.

### Adam-mini (job 497414, W&B `t8tge45y`)

Same settings as wd 0.01 (AdamW β2 0.95), with `--optimizer adam_mini`.

| Optimizer | Final eval loss | Final eval ppl | Eval @100 | Eval @300 | Eval @500 | Eval @700 |
|---|---|---|---|---|---|---|
| **AdamW** (497297) | **4.062** | **58.1** | **6.370** | **5.279** | **4.506** | **4.102** |
| Adam-mini (497414) | 4.331 | 76.0 | 6.564 | 5.637 | 5.001 | 4.379 |

- **Adam-mini is clearly worse (+0.27):** it trails from the start, and the gap peaks at ~0.5 around step 500.
  Same step speed (~2.16 s/it), so it gives no time benefit here either.
- Likely cause: with HF GPT-2's names and Conv1D layout, Adam-mini puts the fused QKV, attention output and
  LayerNorms into one-lr-per-tensor blocks, and gives the MLPs per-input-feature lrs. That is much coarser than the
  head-/neuron-wise partition the method relies on. The LR (tuned for AdamW) may also not suit it.
- Model deleted (W&B keeps the settings and curves).

### Speed tests by batch size (jobs 497502–497505, W&B off)

Median train-step time on 2 GPUs, 32 seqs/GPU, excluding eval; `train.py` now sizes each run as
`(1590 s − 37 evals × 6.4 s) / step time` with 37 evals spread evenly (`--global_batch_size`):

| Global batch | Grad accum | Step time | Steps | Eval every | Tokens |
|---|---|---|---|---|---|
| 64 | 1 | 0.249 s | 5434 | 146 | 356M |
| 128 | 2 | 0.469 s | 2885 | 77 | 378M |
| 256 | 4 | 0.912 s | 1483 | 40 | 389M |
| 512 | 8 | 1.824 s | 741 | 20 | 388M |

Step time is ~linear in batch size, so every run sees ~356–389M tokens; the batch-size test is
"more, noisier updates vs fewer, larger ones" at roughly equal data.

### β2 = 0.999 vs 0.95 (jobs 497488, 497489)

Same settings as above (peak 1.25e-3, WSD 20%, wd 0.01, 746 steps); only β2 and the optimizer differ.

| Optimizer | β2 | Job | W&B run | Final eval loss | Final eval ppl | Eval @500 | Train-loss spikes (> +0.1) |
|---|---|---|---|---|---|---|---|
| **AdamW** | **0.95** | 497297 | `vdvpngw8` | **4.062** | **58.1** | 4.506 | 0 |
| AdamW | 0.999 | 497488 | `c67fz5ud` | 4.322 | 75.3 | 5.174 | 3 (steps 280, 480, 490; up to +0.36) |
| Adam-mini | 0.95 | 497414 | `t8tge45y` | 4.331 | 76.0 | 5.001 | 0 |
| Adam-mini | 0.999 | 497489 | `ojvk029x` | 5.215 | 184.0 | 5.766 | 2 |

- **β2 = 0.999 is much worse for both optimizers** (+0.26 for AdamW, +0.88 for Adam-mini). With β2 = 0.999,
  the second-moment estimate lags far behind at this high LR, so it can't track sudden gradient growth, and
  the result is loss spikes. Keep β2 = 0.95 (the GPT-3/nanoGPT value).
- Adam-mini suffers more, likely because its single second-moment value per large block reacts even more slowly.
- All three losing models deleted.

## 2026-10-05 — Global batch size 64 / 128 / 256 / 512 (AdamW)

**Question:** At a fixed 30-min budget (≈ fixed tokens), are many small updates better than fewer large ones?

### Setup

AdamW, β2 0.95, peak 1.25e-3 (tuned at batch 512, not re-tuned), WSD 20% decay, warmup 10%, wd 0.01,
unfiltered C4. 32 seqs/GPU × 2 GPUs; batch set by grad accum. Steps sized from the speed tests above, with 37 evals
spread evenly.

```
sbatch --job-name=gpt2-c4-bs$B run.sbatch --optimizer=adamw --beta2=0.95 --global_batch_size=$B \
    --learning_rate=1.25e-3 --decay_frac=0.2 --weight_decay=0.01 --run_name=adamw-bs$B-wsd20-lr1.25e-3-wd0.01
```

### Results

| Global batch | Job | W&B run | Steps | Tokens | Final eval loss | Final eval ppl | Final train loss | Max grad norm | Job time | Saved at |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 497557 | `07ikdba5` | 5434 | 356M | 3.789 | 44.2 | 3.857 | 2.74 | 1716 s | 1675 s |
| **128** | 497558 | `esf76yzq` | 2885 | 378M | **3.788** | **44.2** | 3.867 | 2.20 | 1689 s | 1649 s |
| 256 | 497559 | `7myie90p` | 1483 | 389M | 3.833 | 46.2 | 3.918 | 2.09 | 1658 s | 1632 s |
| 512 | 497297 | `vdvpngw8` | 746 | 388M | 4.062 | 58.1 | 4.159 | 1.86 | 1656 s | 1615 s |

Eval loss at the same fraction of the run (≈ same tokens):

| Fraction | bs 64 | bs 128 | bs 256 | bs 512 |
|---|---|---|---|---|
| 10% | **5.450** | 5.730 | 6.082 | 6.521 |
| 25% | **4.441** | 4.541 | 5.064 | 5.769 |
| 50% | **4.115** | 4.140 | 4.267 | 4.916 |
| 75% | 3.981 | **3.978** | 4.045 | 4.353 |
| 90% | 3.851 | 3.851 | 3.891 | 4.126 |
| 100% | 3.789 | **3.788** | 3.833 | 4.062 |

### Findings

- **Smaller batches win big: 3.79 (ppl 44) at batch 64/128 vs 4.06 (ppl 58) at 512, a −0.27 gain**, the
  largest of any change so far. With only ~0.4B tokens, the model benefits far more from 4–7× more optimizer
  updates than from lower-noise gradients. Batch 512 is well above the critical batch size this early in training.
- **64 and 128 tie** (Δ 0.001). Batch 64 leads early (more updates), and 128 catches up by 75%. The gains
  flatten below 128, so the critical batch size here is ≈ 64–128.
- Step time is ~linear in batch size (0.25 → 1.82 s), so GPU efficiency barely drops at small batches; the
  smaller batches are not paying for themselves in lost throughput.
- The LR was not re-tuned for small batches (1.25e-3 from batch 512). Small batches usually prefer a *lower* LR, so 64/128
  may improve further with LR tuning.
- Timing: all finished under the limit, but batch 64 was tightest (1716 s, 84 s spare; its median step 0.251 s
  vs 0.249 s measured). Batch 128 had 111 s spare.

### Models

- Kept **batch 128** (`~/gpt2_models/adamw-bs128-wsd20-lr1.25e-3-wd0.01`) as the new best; it ties batch 64 and has more time margin.
- Deleted batch 64, 256 and the old 512 best (the 512 model is still the Hub's latest version, `7945c6fc31`).
- **Pushed** batch 128 with `push_model.py`: commit `8d130f6417`, sha256 verified. `train.py` default optimizer
  switched back to `adamw` (`--optimizer adam_mini` still available).

### Next

- LR sweep at batch 128 (e.g. 6e-4, 9e-4, 1.25e-3, 1.75e-3).
- Batch 96 or a per-GPU batch change if the critical batch matters; also test whether 64 fits the
  budget more safely with a slightly lower step count.

## 2026-10-05 — Throughput: profiling and fixes (B: fused GELU, C: 24 CPUs, D: 64/GPU)

**Problem:** GPU "Tensor HMMA Active" ~20%; estimated MFU 12% at batch 128 (279k tokens/s).

### Profile (`profile.sbatch`, torch.profiler on steps 9–13, batch 128)

Profiles 499745 (1 CPU) / 499746 (24 CPUs), before fixes: GPU busy (union of kernels) 91% / 94%, so the GPU
was **not starved by the CPU**. But only **40% of GPU time was matmuls** (tensor cores); ~54% was memory-bound
kernels: generic elementwise 29% (GPT-2's `gelu_new`, written in Python as ~8 kernels), copy/cast 15% (autocast
weight casts on every micro-batch), and 6% softmax for the 50k-vocab loss. With 1 CPU, NCCL all-reduce took
235 ms vs 44 ms (its proxy thread competes for the core).

### Fixes, applied one at a time (profile jobs 499799–499801, 40 steps, median over 35)

| Config (batch 128) | Step time | Tokens/s | MFU |
|---|---|---|---|
| Before (1 CPU, `gelu_new`, 32/GPU × accum 2) | 0.469 s | 279k | 12.1% |
| + C: `--cpus-per-task=24` | 0.436 s | 300k | 13.1% |
| + B: `activation_function="gelu_pytorch_tanh"` | 0.317 s | 414k | 18.0% |
| + D: 64/GPU, no grad accum | **0.307 s** | **427k** | **18.6%** |

After the fixes (499801): matmul share 40% → 56%, elementwise 610 → 137 ms per 5 steps. Remaining non-matmul
time: copy/cast 11%, attention 9%, softmax/loss 8.5%. B is numerically the same formula as `gelu_new` (the same tanh
approximation); C and D don't change the math (D only changes reduction order).

### New step times → more steps per 30-min run (speed jobs 499817–499819)

| Global batch | Old s/step | New s/step | Speedup | Steps | Tokens |
|---|---|---|---|---|---|
| 64 | 0.249 | 0.161 | 1.55× | 5434 → 8404 | 356M → 551M |
| 128 | 0.469 | 0.307 | 1.53× | 2885 → 4407 | 378M → 578M |
| 256 | 0.912 | 0.606 | 1.50× | 1483 → 2233 | 389M → 585M |
| 512 | 1.824 | 1.208 | 1.51× | 741 → 1120 | 388M → 587M |

`SEC_PER_STEP` in `train.py` updated. New flags: `--activation` (default `gelu_pytorch_tanh`),
`--per_device_batch` (default 64); `train.py` logs `tokens_per_sec` and `mfu` every 10 steps.

### Next

- A full batch-128 run with the new speed to confirm the timing and measure the loss gain (~1.5× tokens).
- Remaining speedups: `torch.compile` (casts, residual adds, dropout), a fused/chunked cross-entropy for the 50k vocab,
  `adamw_torch_fused`; dropout 0.1 → 0 (a hyperparameter change).

## 2026-10-05 — torch.compile: small gain after B/C/D

**Question:** After the fused GELU, 24 CPUs and 64/GPU, does `torch.compile` (new flag `--torch_compile`) still pay off?

Profile job 499865 (`profile.sbatch --max_steps=60 --torch_compile`, batch 128, eval off) vs 499801 (same without compile):

| | Without compile (499801) | With compile (499865) |
|---|---|---|
| Median step time (excl. eval) | 0.307 s | **0.285 s (−7%)** |
| Step 1 (includes compilation) | 0.7 s | **32.4 s** |
| First step done (from script start) | ~9 s | 42 s |

**Net for a 30-min run:** training budget after evals ≈ 1353 s → without compile 1353 / 0.307 = **4407 steps**;
with compile (1353 − 32) / 0.285 ≈ **4635 steps (+5%)**. The first eval likely triggers a second compilation
(eval mode, different batch size) that this profile couldn't see (eval off), which would eat into the +5%.

### Findings

- **Interesting: compile gives only 7% once the obvious memory-bound kernels are gone.** Most of the
  fusion it would have done (the ~8-kernel `gelu_new` chain, the extra autocast casts from grad accumulation)
  was already removed by B and D. What's left (matmuls 56%, sdpa attention, the 50k-vocab softmax,
  some casts) is either already fused or not something compile changes much.
- Fixed cost of ~32 s compile vs ~7% faster steps → break-even after ~7.5 min of training; in a 30-min
  job the net gain is ~5% more steps, before the probable eval recompile.
- **Decision: leave compile off for now** (marginal gain, extra risk of recompiles and fragility with 2 GPUs).
  Revisit if the remaining kernels change (e.g. a fused cross-entropy), or with a test that has eval on to measure
  the recompile cost.

## 2026-10-05 — Full run with the throughput fixes (batch 128, 4407 steps)

Same hyperparameters as the previous best (AdamW, batch 128, peak 1.25e-3, WSD 20%, wd 0.01, β2 0.95, dropout 0.1),
now with B/C/D (fused GELU, 24 CPUs, 64/GPU), so the 30-min budget fits 4407 steps instead of 2885.

```
sbatch --job-name=gpt2-c4-bs128-fast run.sbatch --decay_frac=0.2 --run_name=adamw-bs128-wsd20-fast
```

| Run | Job | W&B run | Steps | Tokens | Final eval loss | Final eval ppl | Final train loss | Tokens/s | MFU |
|---|---|---|---|---|---|---|---|---|---|
| Previous best | 497558 | `esf76yzq` | 2885 | 378M | 3.788 | 44.2 | 3.867 | ~279k | ~12% |
| **Fast** | 499864 | `lztnhd8g` | 4407 | 578M | **3.677** | **39.5** | 3.753 | 424k | 18.5% |

- **New best: 3.677 (−0.111)**, purely from the speedup (1.53× more steps/tokens in the same 30 min).
- Timing: median step 0.308 s (matches the 0.307 s measurement). Saved at 1474 s, job total 1510 s, so there was
  **~290 s spare**. Evals now take 2.47 s each instead of 6.4 s (the fused GELU also speeds up eval), and startup was ~9 s
  vs the 60 s budgeted. Updating `EVAL_SEC` and the startup reserve would give roughly +10–15% more steps.
- Model kept at `~/gpt2_models/adamw-bs128-wsd20-fast`; the previous best's local copy was deleted. Not pushed yet.

## 2026-10-05 — Dropout 0.1 → 0 (full run)

Identical to the fast run above (499864) except `--dropout=0` (GPT-2's `resid_pdrop`, `attn_pdrop`, `embd_pdrop`).

```
sbatch --job-name=gpt2-c4-bs128-drop0 run.sbatch --decay_frac=0.2 --dropout=0 --run_name=adamw-bs128-wsd20-drop0
```

| Dropout | Job | W&B run | Final eval loss | Final eval ppl | Final train loss | Eval − train | Median step | Tokens/s |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 499864 | `lztnhd8g` | 3.677 | 39.5 | 3.753 | −0.076 | 0.308 s | 424k |
| **0** | 500119 | `lfovx9j2` | **3.609** | **36.9** | **3.620** | −0.011 | 0.294 s | 444k |

Eval loss at the same fraction of the run:

| Fraction | Dropout 0.1 | Dropout 0 | Δ |
|---|---|---|---|
| 10% | 5.320 | 5.243 | −0.077 |
| 25% | 4.270 | 4.213 | −0.057 |
| 50% | 3.971 | 3.921 | −0.050 |
| 75% | 3.844 | 3.794 | −0.050 |
| 90% | 3.727 | 3.667 | −0.060 |
| 100% | 3.677 | 3.609 | −0.068 |

### Findings

- **Dropout 0 is better: 3.609 (ppl 36.9) vs 3.677 (−0.068), the new best.** It leads at every point of the run by
  0.05–0.08. A single-seed gap of this size is near the noise level measured earlier, but a lead that
  never flips across the whole run makes it convincing.
- Why: the model sees < 0.2 epoch, so it can't overfit. Dropout only adds noise and slows learning. With
  dropout 0.1, eval loss was *below* train loss (−0.076, because train loss is measured with dropout on); with
  dropout 0 the two almost match (−0.011). This matches nanoGPT and most modern pretraining, which use 0.
- Bonus: 4.5% faster (0.294 vs 0.308 s/step, 444k tokens/s). The run used the 0.307 s budget, so it finished
  even earlier (1449 s total).
- Model kept at `~/gpt2_models/adamw-bs128-wsd20-drop0`; the dropout 0.1 model was deleted. Not pushed yet.

### Follow-up (done)

- **Pushed** the dropout-0 model (`push_model.py`): commit `dde30a2812`, sha256 verified.
- **Defaults are now the best setting:** `--dropout 0`, `--decay_frac 0.2` (with AdamW, batch 128, peak 1.25e-3,
  wd 0.01, β2 0.95, fused GELU, 64/GPU), so a plain `sbatch run.sbatch --run_name=...` reproduces it.
- **Time budget refit from job 500119:** startup reserve 60 → 30 s, wrap-up 30 → 45 s (34 s measured outside the
  script), `EVAL_SEC` 6.4 → 2.6 s, batch-128 step time 0.307 (median) → 0.298 s (mean over a full run, incl. logging).
  `TIME_BUDGET` 1590 → 1605 s → **batch 128: 4407 → 5063 steps (+15%)**; predicted job total ~1649 s (151 s spare;
  training ends ~1614 s, before the 1710 s deadline stop). Not yet confirmed with a full run.

## 2026-10-06 — Memory headroom, and padding the vocab 50257 → 50304

**Questions:** (1) Does 64 seqs/GPU fill the H200's memory? (2) Is training compute-bound?

### Memory (profile jobs 503405–503407, grad accum 1, peak since process start)

| Per GPU (global batch) | Peak allocated | Peak reserved | Share of 139.8 GiB | Step time (profiler on) |
|---|---|---|---|---|
| **64 (128) — default** | 59.5 GiB | 65.2 GiB | **47%** | 0.348 s |
| 128 (256) | 116.9 GiB | 129.2 GiB | 92% | 0.623 s |
| 256 (512) | OOM | — | — | — |

- ~0.9 GiB per sequence; the ceiling is ~135 seqs/GPU. With global batch 128 on 2 GPUs, 64/GPU is already the cap.
- **The fp32 logits use most of the memory, not the model.** The 256/GPU run OOMed allocating 49.08 GiB =
  256 × 1024 × 50257 × 4 bytes; weights + grads + Adam state for 124M params are ~2 GiB.
- 128/GPU at batch 256 (0.623 s, profiler on) gives no real gain over 64/GPU × accum 2 (0.606 s): bigger
  micro-batches don't help, so 64/GPU stays the default.

### Compute-bound? No: one slow GEMM

- GPU busy ~87% (kernel time / wall time), so the GPU is not starved, but MFU was only ~18.6%.
- **~40% of GPU time went to three `cutlass_75_tensorop_bf16_s1688gemm` kernels** (251 + 179 + 166 ms per 5 steps,
  one call each per step = the LM head forward + 2 backward GEMMs). This is a Turing-era (sm75) fallback: the
  odd vocab size 50257 rules out the Hopper kernels; all the other GEMMs use `nvjet_sm90`.
- **Fix: `--pad_vocab`** pads `vocab_size` to 50304 (a multiple of 128; the nanoGPT trick). The 47 extra token IDs never
  appear in the data; their logits are just pushed down. Before saving, `resize_token_embeddings(50257)` trims the
  tied wte/lm_head back so the saved model is a standard GPT-2.

Profiles at 64/GPU, batch 128 (503420 vs 503421; 503423 repeats the padded run):

| | Step time (profiler on) | GPU kernel time / 5 steps | GPU busy | Peak reserved |
|---|---|---|---|---|
| Unpadded | 0.341 s | 1.477 s | 87% | 65.2 GiB |
| **Padded** | **0.240 s** (0.242 repeat) | 0.966 s | 80% | 65.2 GiB |

The sm75 kernels are gone, and the LM head runs on `nvjet_sm90`. GPU busy drops slightly because GPU work shrank
while launch/all-reduce overhead stayed the same.

**Profiler fix:** the "GPU busy" line printed 296% because it summed `self_device_time_total` over all events,
counting kernels again through the CPU ops that launch them (`aten::mm`) and the GPU-side user annotations
(`ProfilerStep*`, `DistributedDataParallel.forward`). It now sums only CUDA events that aren't user annotations,
the same rule as the table's "Self CUDA time total" footer (verified equal in 503423: 0.973 s). `profile.sbatch`
also logs fb memory used (`nvidia-smi dmon -s um`); `dmon`'s `mem%` is memory *bandwidth*, not how full the GPU is.

### Step times with padding (speed jobs 503436–503440, 60 steps, median of 55)

| Global batch | Unpadded s/step (SEC_PER_STEP) | Padded median | New SEC_PER_STEP | Steps per 30-min run |
|---|---|---|---|---|
| 64 | 0.161 | 0.104 | 0.106 | 8404* → 14233 |
| 128 | 0.298 | 0.195 (unpadded control 503440: 0.293) | 0.198 | 5063 → **7620** |
| 256 | 0.606 | 0.386 | 0.393 | 2233* → 3839 |
| 512 | 1.208 | 0.757 | 0.770 | 1120* → 1959 |

\* before the time-budget refit. **1.50× faster at batch 128** (0.293 → 0.195 s median). New `SEC_PER_STEP` =
padded median × 1.017 (full-run mean 0.298 / speed-job median 0.293, both unpadded, to cover logging overhead).
`--pad_vocab` (now `BooleanOptionalAction`, `--no-pad_vocab` disables) is on by default in `run.sbatch` and `profile.sbatch`.

### Full runs (batch 128, defaults: AdamW, peak 1.25e-3, WSD 20%, wd 0.01, β2 0.95, dropout 0)

```
sbatch --job-name=gpt2-c4-padvocab-4407 run.sbatch --max_steps=4407 --run_name=adamw-bs128-wsd20-padvocab-4407
sbatch --job-name=gpt2-c4-padvocab      run.sbatch --run_name=adamw-bs128-wsd20-padvocab
```

| Run | Job | Vocab | Steps | Tokens | Final eval loss | Final eval ppl | Final train loss | Median step | Tokens/s | MFU | Job total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (dropout 0) | 500119 | 50257 | 4407 | 578M | 3.609 | 36.93 | 3.620 | 0.294 s | 445k | 19.4% | 1449 s |
| Loss check | 503441 | 50304 | 4407 | 578M | 3.608 | 36.89 | 3.618 | 0.198 s | 660k | 28.7% | 989 s |
| **Full budget** | 503448 | 50304 | **7620** | **999M** | **3.513** | **33.54** | 3.514 | 0.196 s | 667k | 29.0% | 1616 s |

Loss check, eval loss (ppl) at the same step (500119 vs 503441): 6.481 / 6.478 (step 119), 4.276 (71.9) / 4.277 (72.0) (1071),
3.830 / 3.829 (2975), 3.692 / 3.691 (3927), 3.609 (36.93) / 3.608 (36.89) (end). Largest gap: +0.058 at step 595 (one point, early).

Full budget vs baseline at the same fraction of the run, eval ppl (loss):

| Fraction | Baseline 500119 | Full budget 503448 | Δ ppl |
|---|---|---|---|
| 11% | 189.2 (5.243) | 102.6 (4.630) | −86.6 |
| 24% | 71.9 (4.276) | 56.8 (4.040) | −15.1 |
| 49% | 51.5 (3.941) | 44.2 (3.789) | −7.2 |
| 76% | 44.4 (3.794) | 39.5 (3.676) | −4.9 |
| 100% | 36.93 (3.609) | **33.54 (3.513)** | **−3.39** |

### Findings

- **Padding doesn't change the learning curve:** at the same steps the loss matches the unpadded baseline to ±0.003
  for nearly the whole run (final ppl 36.89 vs 36.93, loss 3.608 vs 3.609). The 47 unused logits cost nothing measurable.
- **New best: ppl 33.54 (loss 3.513), −3.39 ppl (−9.2%) vs 36.93 (3.609)**, purely from the speedup: 1.73× more steps/tokens (7620 vs 4407)
  in the same 30 minutes. It leads at every point of the run.
- **MFU 19.4% → 29.0%.** The run finished at 1616 s (184 s spare). Mean step ≈ (1578 s − 38 evals × ~2.5 s) / 7620
  ≈ 0.195 s, so `SEC_PER_STEP[128]` = 0.198 is ~1.5% conservative.
- The saved model has `vocab_size` 50257 and `wte` [50257, 768] (trimmed back), 498 MB like before.
- Kept `~/gpt2_models/adamw-bs128-wsd20-padvocab` (503448); deleted the 500119 model (already on the Hub) and the
  503441 loss-check model. Not pushed yet.

### Next

- Remaining non-matmul time: fused/chunked cross-entropy (fp32 logits are also ~most of the memory), copy/cast,
  `adamw_torch_fused`. Retry `torch.compile` now that the LM head isn't the bottleneck.
- Freed memory could allow a bigger micro-batch, but 128/GPU gave no gain over 64 × accum 2 before padding.

## 2026-10-06 — LR re-scan at batch 128, 7620 steps (padded vocab)

**Question:** Peak LR 1.25e-3 was tuned at batch 512 / 631 steps. Is it still right at batch 128 / 7620 steps?
(Square-root batch scaling suggests ~6.25e-4.)

### Setup

Current defaults (AdamW, batch 128, 7620 steps, warmup 762, WSD 20%, wd 0.01, β2 0.95, dropout 0, `--pad_vocab`);
only `--learning_rate` changes. 1.25e-3 reuses run 503448.

```
for LR in 5e-4 7.5e-4 1e-3 1.6e-3 2e-3; do
  sbatch --job-name=gpt2-c4-lr$LR run.sbatch --learning_rate=$LR --run_name=adamw-bs128-wsd20-lr$LR
done
```

### Results

| Peak LR | Job | W&B run | Final eval ppl | Δ ppl vs 1.25e-3 | Final eval loss | Final train loss | Max grad norm |
|---|---|---|---|---|---|---|---|
| 5e-4 | 503722 | `0kb9jr8j` | 34.31 | +0.77 | 3.535 | 3.536 | 6.56 |
| 7.5e-4 | 503723 | `yw7ei8ow` | 34.09 | +0.55 | 3.529 | 3.530 | 4.78 |
| 1e-3 | 503724 | `f647ie3t` | 33.78 | +0.24 | 3.520 | 3.521 | 4.14 |
| 1.25e-3 | 503448 | — | 33.54 | — | 3.513 | 3.514 | 3.47 |
| **1.6e-3** | 503725 | `64tt99mv` | **33.51** | −0.03 | **3.512** | 3.512 | 2.80 |
| 2e-3 | 503726 | `clr9v4bj` | 33.54 | 0.00 | 3.513 | 3.513 | 2.66 |

Eval ppl at the same point of the run:

| Fraction | 5e-4 | 7.5e-4 | 1e-3 | 1.25e-3 | 1.6e-3 | 2e-3 |
|---|---|---|---|---|---|---|
| 11% | 150.1 | 118.1 | 108.2 | 102.6 | 98.7 | **97.1** |
| 24% | 62.0 | 59.3 | 57.6 | 56.8 | 56.6 | **56.3** |
| 49% | 46.1 | 45.4 | 44.7 | 44.2 | 44.2 | **44.1** |
| 76% | 40.3 | 40.2 | 39.8 | **39.5** | **39.5** | 39.7 |
| 89% (decay starts ~80%) | 36.9 | 36.8 | 36.5 | **36.2** | **36.2** | 36.3 |
| 100% | 34.31 | 34.09 | 33.78 | 33.54 | **33.51** | 33.54 |

### Findings

- **Lower LR is worse, not better:** ppl rises steadily below 1.25e-3 (1e-3: +0.24, 7.5e-4: +0.55, 5e-4: +0.77).
  The square-root batch-scaling guess (~6e-4) would have cost ~0.6 ppl.
- **1.25e-3 to 2e-3 is a flat plateau: 33.51–33.54.** The 0.03-ppl spread is at the noise level, so there's no reason
  to change the default; 1.25e-3 stays. Higher LRs lead early (2e-3 best through 49%), but the lead disappears by the
  end of the decay.
- Higher LR gave *lower* max grad norm (2.7 at 2e-3 vs 6.6 at 5e-4); no spikes or instability at 2e-3, unlike at
  batch 512 / 631 steps, where 2e-3 was clearly too high.
- LR is no longer a lever at this setup; gains have to come from elsewhere (more steps/throughput, model/data).
- Kept `~/gpt2_models/adamw-bs128-wsd20-lr1.6e-3` (nominal best); deleted the other four scan models and the
  local 503448 copy (on the Hub as `7476e47d9b`). Not pushed: 0.03 ppl isn't a real improvement.

## 2026-10-06 — Fused LM head + cross-entropy (Liger), `--fused_ce`

**Question:** Can we drop the full fp32 logits (log_softmax ~13% of GPU time, most of the memory) with an existing kernel?

### Setup

`liger-kernel` 0.8.4 (Triton; torch/triton unchanged), `LigerFusedLinearCrossEntropyLoss` called from a small forward
patch in `train.py` (`--fused_ce`, off by default): `model.transformer` → hidden states, shift as HF does, fused loss with
`lm_head.weight` (tied to `wte`), `sum / num_items_in_batch` when the Trainer passes it, else mean. It computes logits,
loss and gradients chunk by chunk in the forward pass (~1024 tokens per chunk at 64/GPU).

### Correctness (job 504072, `logs/fused_ce_check.py`, 1 GPU, bf16 autocast, same weights and batch 8×1024)

| Case | Loss ref → fused | Grad rel. error (wte / h.0 c_attn / h.11 c_proj / ln_f) | Peak mem |
|---|---|---|---|
| mean | 10.982760 → 10.982761 | 0.4% / 0.6% / 0.15% / 0.02% (cosine ≥ 0.99998) | 8.7 → 5.0 GiB |
| mean, 100 labels/row = -100 | identical | same | 9.5 → 5.3 GiB |
| sum / num_items_in_batch | identical | ≤ 0.7% | 9.5 → 5.3 GiB |

Gradient differences are at bf16 rounding level (the reference also computes the logits in bf16).

### Profile (504062 default vs 504063 `--fused_ce`; both padded vocab, 64/GPU, batch 128, 60 steps)

| | Default | `--fused_ce` |
|---|---|---|
| Median step (55 steps, profiler mostly off) | 0.193 s | **0.156 s (−19%, 1.24×)** |
| GPU kernel time / 5 profiled steps | 1.019 s | 0.777 s (−24%) |
| GPU busy | 86% | 72% |
| Peak allocated / reserved | 59.6 / 65.2 GiB | **23.3 / 23.9 GiB** |
| `log_softmax` fwd + bwd | 127 ms | gone |
| `aten::copy_` (casts) | 193 ms | 65 ms |
| `aten::mm` + `addmm` calls | 975 | 1920 (chunked) |

- The softmax kernels and the large fp32 casts are gone; the matmuls stay the same.
- GPU busy fell to 72%: kernels got shorter, and the chunk loop launches ~2× more matmuls, so CPU launch overhead is
  now visible. That's a candidate for `torch.compile` / CUDA graphs later.
- Memory drops from 65 to 24 GiB reserved. 64/GPU is still the cap at global batch 128 (2 GPUs).

### Next

- Speed jobs (64/128/256/512) with `--fused_ce` → refit `SEC_PER_STEP` (batch 128: ~0.198 → ~0.16 s, ~9.4k steps).
- Full runs: same-step check at 7620 steps vs 503448 (ppl 33.54), then the full new budget.

### Full runs with `--fused_ce` (now on by default in `run.sbatch` / `profile.sbatch`)

Speed jobs 504142–504145 (60 steps, median of 55): batch 64: 0.091 s, 128: 0.156 s, 256: 0.308 s, 512: 0.617 s; peak
reserved 13–24 GiB. `SEC_PER_STEP` = median × 1.017 → {64: 0.093, 128: 0.159, 256: 0.313, 512: 0.627};
**batch 128: 7620 → 9489 steps**.

```
sbatch --job-name=gpt2-c4-fusedce-7620 run.sbatch --max_steps=7620 --run_name=adamw-bs128-wsd20-fusedce-7620
sbatch --job-name=gpt2-c4-fusedce      run.sbatch --run_name=adamw-bs128-wsd20-fusedce
```

| Run | Job | W&B run | Steps | Tokens | Final eval ppl | Δ ppl | Final eval loss | Median step | Tokens/s | MFU | Job total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Padded, no fused CE | 503448 | — | 7620 | 999M | 33.54 | — | 3.513 | 0.196 s | 667k | 29.0% | 1616 s |
| Fused CE, same steps | 504146 | `quf9pjj1` | 7620 | 999M | 33.59 | +0.05 | 3.514 | 0.159 s | 825k | 35.9% | 1340 s |
| **Fused CE, full budget** | 504157 | `s0dqmrl4` | **9489** | **1.24B** | **32.34** | **−1.20 (−3.6%)** | **3.476** | 0.161 s | 815k | 35.5% | 1673 s |

Eval ppl at the same fraction of the run:

| Fraction | 503448 (7620) | 504146 (fused, 7620) | 504157 (fused, 9489) |
|---|---|---|---|
| 11% | 102.6 | 101.7 | 90.5 |
| 24% | 56.8 | 57.3 | 52.6 |
| 49% | 44.2 | 44.5 | 41.9 |
| 76% | 39.5 | 39.6 | 37.9 |
| 89% | 36.24 | 36.29 | 34.82 |
| 100% | 33.54 | 33.59 | **32.34** |

### Findings

- **Same quality per step:** at 7620 steps, fused CE ends at ppl 33.59 vs 33.54 (+0.05, the size of the LR-scan spread
  between 1.25e-3 / 1.6e-3 / 2e-3). Mid-run gaps reached +0.43 ppl (24%) and shrank to the end, consistent with bf16
  rounding differences changing the trajectory slightly, not a systematic loss.
- **New best: ppl 32.34 (loss 3.476), −1.20 ppl vs 33.54**, from 1.25× more steps (9489 vs 7620) in the same 30 min.
  It leads at every point of the run.
- MFU 29% → 35.5%; 815k tokens/s.
- **The time budget is now tight:** full-run mean step = (1628 s − 71 s eval) / 9489 = 0.164 s, ~3% above the 0.159 used
  for sizing (median 0.161). Training ended at 1636 s (deadline stop 1710 s), job total 1673 s, 127 s spare. Evals got
  cheaper (1.88 s vs 2.6 s budgeted), which partly offsets it. Refitting both (0.164 s, 1.9 s) would give ~9360 steps.
- Kept `~/gpt2_models/adamw-bs128-wsd20-fusedce` (504157); deleted the 504146 check model and the LR-scan 1.6e-3 model
  (ppl 33.51). Pushed to the Hub: commit `268ae0e8e8` (now the latest version). Code: git `5b4d24e`.

## 2026-10-06 — Warmup / decay length at 9489 steps (AdamW, fused CE)

**Question:** Warmup (10%) and decay (20%) were set at 631 steps. With 15× more steps, are they still right?

### Setup

Current defaults (AdamW 1.25e-3, batch 128, 9489 steps, WSD, padded vocab, fused CE); new flag `--warmup_frac`
(default 0.1). Baseline = 504157 (warmup 10%, decay 20%, ppl 32.34). One factor at a time:

```
sbatch --job-name=gpt2-c4-wu$W-dc$D run.sbatch --warmup_frac=$WF --decay_frac=$DF --run_name=adamw-bs128-wu$W-wsd$D-fusedce
```

### Results

| Warmup | Decay | Job | W&B run | Final eval ppl | Δ ppl | Final eval loss | Final train loss | Max grad norm |
|---|---|---|---|---|---|---|---|---|
| **2% (189)** | 20% | 504449 | `a3obmncq` | **30.87** | **−1.47** | **3.430** | 3.446 | 2.16 |
| 5% (474) | 20% | 504450 | `azhsro0w` | 31.36 | −0.98 | 3.446 | 3.463 | 2.64 |
| 10% (948) | 20% | 504157 | `s0dqmrl4` | 32.34 | — | 3.476 | 3.494 | 3.83 |
| 20% (1897) | 20% | 504458 | `2xuiqmcc` | 32.15 | −0.19 | 3.471 | 3.486 | 6.29 |
| 10% | 10% | 504451 | `4i0w9qpw` | 32.70 | +0.36 | 3.487 | 3.505 | 3.79 |
| 10% | 30% | 504452 | `kj2mt60a` | 32.27 | −0.07 | 3.474 | 3.491 | 3.83 |
| 10% | 40% | 504453 | `mp04wkg2` | 32.18 | −0.16 | 3.471 | 3.488 | 4.00 |

Eval ppl at the same step (eval every 256 steps):

| Step | wu 2% | wu 5% | wu 10% (base) | wu 20% | dc 10% | dc 30% | dc 40% |
|---|---|---|---|---|---|---|---|
| 256 | **330.5** | 407.8 | 456.1 | 549.1 | 466.5 | 457.5 | 451.2 |
| 1024 | **70.6** | 75.3 | 90.5 | 106.1 | 88.5 | 88.9 | 88.2 |
| 2304 | **49.1** | 50.4 | 52.6 | 57.3 | 52.8 | 52.6 | 52.5 |
| 4608 | **40.0** | 40.7 | 41.9 | 42.8 | 42.0 | 41.9 | 41.8 |
| 7168 | 36.2 | 36.7 | 37.9 | 37.9 | 37.9 | 37.0 | **35.9** |
| 8448 | **33.2** | 33.8 | 34.8 | 34.7 | 36.7 | 33.9 | 33.5 |
| 9489 (end) | **30.87** | 31.36 | 32.34 | 32.15 | 32.70 | 32.27 | 32.18 |

### Findings

- **Shorter warmup is the big lever: 2% → ppl 30.87, −1.47 vs 10%**, the new best. 5% is in between (−0.98). The
  2% run leads at every eval of the run, and also has the lowest max grad norm (2.16 vs 3.8), so the short warmup
  isn't destabilizing anything.
- Longer warmup doesn't help: 20% ends at 32.15 vs 32.34 for 10% (−0.19, about the noise level, and behind 10%
  until the decay). The 10% default made sense at 631 steps (63 warmup steps), but at 9489 steps it spends ~950 steps
  below peak LR.
- **Decay length matters much less:** 20% / 30% / 40% end at 32.34 / 32.27 / 32.18, a slight trend toward longer
  decay that's within ~0.15 ppl; 10% is clearly worse (+0.36).
- **Extension below 2%** (same setup, decay 20%):

  | Warmup | Job | W&B run | Final eval ppl | Δ vs 10% | Final eval loss | Max grad norm | Eval ppl @256 / 1024 / 4608 |
  |---|---|---|---|---|---|---|---|
  | 0.5% (47) | 504661 | `n6vbwx4k` | 31.92 | −0.42 | 3.463 | 2.02 | 353.0 / 79.2 / 41.5 |
  | **1% (94)** | 504660 | `zl176mxp` | **30.77** | **−1.57** | **3.427** | 2.29 | 311.6 / 70.4 / 39.7 |
  | 2% (189) | 504449 | `a3obmncq` | 30.87 | −1.47 | 3.430 | 2.16 | 330.5 / 70.6 / 40.0 |

  **The optimum is 1–2% warmup** (30.77 vs 30.87, within noise). 0.5% is clearly worse (+1.15 vs 1%), behind from the
  first eval: too few warmup steps hurt early, and it never catches up.
- Pushed 504449 to the Hub: commit `689d3f25b7` (now the latest version). Deleted the other schedule-scan models and
  504157's local copy (on the Hub as `268ae0e8e8`). Later kept 504660 (1%, nominal best, not pushed) and deleted the
  504449 local copy (on the Hub) and the 0.5% model.

## 2026-10-06 — Muon (modded-nanogpt style) for the block matrices, first LR scan

**Question:** Does Muon on the transformer blocks' 2D weights beat AdamW at the same 30-min budget?

### Setup

New `muon_adamw.py`: `torch.optim.Muon` (Nesterov momentum 0.95, 5 Newton–Schulz steps in bf16, decoupled wd 0.01,
`adjust_lr_fn="match_rms_adamw"` because HF's Conv1D weights are (in, out)) for `h.*.attn.c_attn / attn.c_proj /
mlp.c_fc / mlp.c_proj` (48 tensors, 84.9M params); AdamW (1.25e-3, β 0.9/0.95, no decay on 1D params) for
`wte` (tied), `wpe`, biases and LayerNorms (100 tensors, 39.5M). A small `MuonWithAdamW` wrapper exposes both
optimizers' param groups, so the Trainer's WSD scheduler drives both LRs. Flags: `--optimizer muon --muon_lr --muon_momentum`.

- Check (job 504419, `logs/muon_check.py`, 1 GPU): split as intended; both groups follow the WSD schedule; loss falls
  10.97 → 6.4 in 60 steps, no NaNs; `opt.step` 14 ms.
- Speed (504422, 2 GPUs, batch 128): median step 0.172 s vs 0.156 s AdamW (+10%); full-run mean 0.172 vs 0.164 (+5%).
  `Muon.step` runs per tensor (many small launches, GPU busy 59%), so it's CPU-launch bound.

### Results (warmup 10%, decay 20%; sized with `--sec_per_step=0.180` → 8382 steps, ~110 s shorter than AdamW's runs)

| Muon LR | Job | W&B run | Final eval ppl | Final eval loss | Max grad norm |
|---|---|---|---|---|---|
| **2.5e-3** | 504430 | `mhaqljmq` | **33.61** | 3.513 | 11.4 |
| 5e-3 | 504431 | `8xlmyjvu` | 36.14 | 3.588 | 1.85 |
| 1e-2 | 504432 | `5sobn4fq` | 886.9 (diverged) | — | 470 |
| 2e-2 | 504434 | `zsv0xp7r` | 1066 (diverged) | — | 1949 |
| AdamW (504157, 9489 steps) | | | 32.34 | 3.476 | 3.83 |

Muon 2.5e-3 vs AdamW by step: ~900: 59.8 vs 90.5 (AdamW @1024); ~1800: 46.9 vs 55.6 (@2048); ~4300: 37.9 vs 41.4 (@4864).

### Findings

- **Muon learns much faster per step early on**, but ends behind (33.61 vs 32.34): it had 12% fewer steps, and it
  gains far less from the final decay (~1.5 ppl over the decay vs ~5.5 for AdamW).
- **1e-2 and 2e-2 diverged during warmup** (loss back up to 6.6–7.0, grad norm 50–2000). Gradient clipping doesn't
  bound Muon's step (orthogonalization removes the gradient's scale), so the LR is the only limit. modded-nanogpt's
  LR (~9e-3 in these units) is too high for plain GPT-2 without its stabilizers (QK-norm etc.).
- 5e-3 trails 2.5e-3 from step ~900 on, so the optimum is ≤ 2.5e-3.

### Equal-time runs (`--sec_per_step=0.167` = Muon's mean × AdamW's sizing ratio → 9034 steps; warmup 10%, decay 20%)

| Muon LR | Job | W&B run | Final eval ppl | Δ vs AdamW 32.34 | Final eval loss | Max grad norm |
|---|---|---|---|---|---|---|
| **1.25e-3** | 504612 | `7kz8u69o` | **32.88** | +0.54 | 3.493 | 4.74 |
| 1.75e-3 | 504637 | `5mwkl2s6` | 32.93 | +0.59 | 3.494 | 7.14 |
| 2.5e-3 | 504614 | `ih5bf1qd` | 33.53 | +1.19 | 3.512 | 3.55 |

Eval ppl at evals 1 / 4 / 9 / 18 / 28 (~238 steps apart) for Muon 1.25e-3: 180.1 / 58.2 / 44.6 / 37.7 / 34.5, vs AdamW
(256 apart): 456.1 / 90.5 / 52.6 / 41.9 / 37.9.

### Findings (equal time)

- **Muon doesn't beat AdamW here: best 32.88 vs 32.34 (+0.54)** at the same warmup/decay and equal time. It's far ahead
  for most of the run, but AdamW's final decay closes the gap and overtakes it. 1.25e-3 and 1.75e-3 are tied, so a lower
  LR is unlikely to help much.
- Both Muon runs hit the 30-min limit (`TIMEOUT`) *after* saving at ~1617 s: something during shutdown (W&B finish /
  process teardown) hung for > 3 min. Models and logs were complete.
- Not tried yet: Muon with the 1–2% warmup (it used 10%), separate Q/K/V orthogonalization, momentum warmup, batched /
  sharded Newton–Schulz to cut the +17 ms of GPU idle per step (profile 504422).
- Deleted the three Muon models.
- Deleted the four scan models.
- Job 504613 (1.75e-3) died at startup on **25a-hgpn157**: other processes held 137.7 GiB of each GPU. Resubmitted as
  504637 with that node excluded.

## 2026-10-06 — Muon with the 1% warmup

Same as 504612 (Muon 1.25e-3, AdamW part 1.25e-3, equal time, 9034 steps) but `--warmup_frac=0.01` (now the default).

| Run | Job | W&B run | Final eval ppl | Eval ppl @ evals 1 / 9 / 28 / 33 |
|---|---|---|---|---|
| AdamW, warmup 1% | 504660 | `zl176mxp` | **30.77** | 311.6 / 48.9 / 36.1 / **33.1** |
| Muon, warmup 1% | 504763 | `08yakwlz` | 32.79 | **182.1 / 44.3 / 34.4** / 33.5 |
| Muon, warmup 10% | 504612 | `7kz8u69o` | 32.88 | 180.1 / 44.6 / 34.5 / 33.5 |

- **Warmup barely matters for Muon** (−0.09): its update size is set by the LR, not the gradient scale, so the short
  warmup that helped AdamW by 1.6 ppl does nothing for it. The gap to AdamW widens to **+2.0 ppl**.
- Muon still leads for ~80% of the run and loses it all in the final decay (AdamW 33.1 → 30.77, Muon 33.5 → 32.79).
- **Stopped Muon here.** `--optimizer muon` stays as an option. Untried: longer decay for Muon, momentum warmup,
  separate Q/K/V orthogonalization, batched Newton–Schulz.

## 2026-10-06 — Profiler fix: real GPU idle is ~5%, not 9%

`ProfilerCallback` called `torch.cuda.synchronize()` after **every** step, so the CPU couldn't queue the next step and
the batch-loading gap at each step start (`aten::ne` on the labels, 4.5 ms/step on average) appeared as GPU idle. Fixed:
sync only at the two window edges; window timed before `prof.step()` collects the trace (that added ~0.25 s per
window, which made the printed busy % read low). New flags: `--profile_start` (profile steps start+1..start+5, also
inside a full run), `--profile_cpu` / `--no-profile_cpu` (GPU kernels only), `--eval` / `--no-eval` (`profile.sbatch`
passes `--no-eval`; `--profile_dir` no longer disables eval).

| Profile (default setup, steps 31–35) | Idle in trace | Printed GPU busy |
|---|---|---|
| Old, sync every step (504143) | 14.7 ms/step (8.9%) | 74% (wrong window timing) |
| Fixed, CPU + GPU (504818 / 504820) | 9.0–9.6 ms/step (5.5–5.9%) | 96% (504820) |
| Fixed, GPU only (504819) | 7.8 ms/step (4.7%) | 85% (old window timing) |

Consistent with real runs: kernels need ~151 ms/step, the unprofiled median step is 156 ms.

A larger per-GPU batch can't fill this: per-sequence time is flat at batch 128 / 256 / 512 with fused CE
(0.00122 / 0.00120 / 0.00121 s), and more sequences per GPU means a larger global batch (worse sample efficiency).

## 2026-10-06 — `torch.compile` on the transformer blocks

**Question:** After padding and fused CE, ~37% of GPU time is memory-bound (LayerNorm 15.7, casts 13.0, elementwise
10.9, GELU 7.0 ms/step, profile 504143). Can compile fuse it?

### Profiles (60 steps, steps 31–35 profiled, default setup)

| Variant | Job | Median step | Attention | LN + casts + elementwise | Step-1 time |
|---|---|---|---|---|---|
| No compile | 504820 | 0.157 s | ~14 ms | 40.4 ms | — |
| `model.transformer.compile()` | 504827 | 0.162 s | **38.5 ms** | 17.9 ms | 20.6 s |
| **Each block `.compile()`** | 504833 | **0.140 s (1.12×)** | 15.7 ms | 18.7 ms | 6.9 s |

Compiling the whole `model.transformer` made the same cuDNN attention kernels 2.5–2.8× slower (same kernel names and
call counts), most likely from how HF's causal-mask setup is traced (an explicit mask instead of `is_causal`); not
confirmed. Compiling only the blocks keeps the mask setup in eager mode and gets the fusion. `--torch_compile` now
compiles each block (`block.compile()`; one compile, reused by all 12 blocks), not the whole model via the Trainer.

### Full runs (warmup 1%, otherwise defaults)

```
sbatch --job-name=gpt2-c4-compile-9489 run.sbatch --torch_compile --max_steps=9489 --run_name=adamw-bs128-wu1-wsd20-compile-9489
sbatch --job-name=gpt2-c4-compile      run.sbatch --torch_compile --sec_per_step=0.147 --run_name=adamw-bs128-wu1-wsd20-compile
```

| Run | Job | W&B run | Steps | Final eval ppl | Δ ppl | Final eval loss | Median / mean step | MFU |
|---|---|---|---|---|---|---|---|---|
| Eager | 504660 | `zl176mxp` | 9489 | 30.77 | — | 3.427 | 0.159 / 0.164 s | ~35% |
| Compiled, same steps | 504862 | `65yuxsg2` | 9489 | 31.14 | +0.37 | 3.438 | 0.143 s | |
| **Compiled, full budget** | 504863 | `v2bjgk57` | **10263** | **30.43** | **−0.34** | **3.416** | 0.143 / 0.145 s | **39.9%** |

Same-step eval ppl, eager vs compiled: eval 4: 70.37 / 71.18; 9: 48.87 / 49.43; 18: 39.74 / 40.34; 28: 36.05 / 36.43;
38: 30.77 / 31.14.

### Findings

- **New best: ppl 30.43 (−0.34)** from 8% more steps (10263 vs 9489). The step-1 compile cost is ~4 s and evals
  didn't trigger slow recompiles (job total ~26.5 min).
- **The same-step run is +0.37 ppl worse, and the gap doesn't close** (unlike fused CE's same-step check, +0.43 mid-run
  → +0.05 at the end). Either run-to-run noise (one seed each; kernel rounding changes the trajectory), or a small
  real numeric difference in the compiled blocks. Not resolved: needs a second seed of eager and compiled.
- `--sec_per_step=0.147` was conservative: mean 0.145 s.
- Kept `~/gpt2_models/adamw-bs128-wu1-wsd20-compile` (504863); deleted 504660 and 504862. Not pushed yet.

## 2026-10-06 — FlashAttention 3 / 4 vs cuDNN SDPA

New flag `--attn_implementation` (default `sdpa`). FA3 = HF's fallback Hub kernel `kernels-community/vllm-flash-attn3`
(`kernels` 0.17.2; FA3 has no PyPI wheel); FA4 = `flash-attn-4` 4.0.0b33 (CuTe DSL, sm90 kernels). Torch unchanged.
Profiles 504895–504898 (60 steps, default setup):

| Backend | Eager median step | Compiled median step | Attention ms/step (eager / compiled) | Step 1 (compiled) |
|---|---|---|---|---|
| SDPA (cuDNN flash) | 0.157 s | **0.140 s** | ~14 / 15.7 | 6.9 s |
| FlashAttention 3 | 0.158 s | 0.146 s | 15.6 / 15.9 | 12.5 s |
| FlashAttention 4 | **0.156 s** | 0.143 s | 15.5 / 15.8 | 16.6 s |
| FlashAttention 2 (Hub kernel `kernels-community/flash-attn2`, 505020 / 505021) | 0.164 s | 0.151 s | 24.5 / 24.5 | 10.7 s |

- **No gain:** at head dim 64 and sequence 1024 on the H200, cuDNN's flash attention is already as fast as FA3 / FA4,
  and attention is only ~10% of the step. Kept SDPA; no full runs (same step time → same steps → same expected ppl).
- **FA2 is slower** (attention 24.5 vs ~15 ms/step, +8% step time): it predates Hopper and doesn't use its
  wgmma / TMA instructions, while cuDNN, FA3 and FA4 all run sm90-specific kernels.
- Kept `sdpa` as the default.

## 2026-10-06 — New defaults: `--torch_compile` on; best model pushed

- `run.sbatch` / `profile.sbatch` now pass `--torch_compile` (`--no-torch_compile` disables).
- `SEC_PER_STEP[128]` = 0.146 (504863 full-run mean 0.145 + ~1%) → **batch 128: 10334 steps**. 64 / 256 / 512 are
  the no-compile values scaled by 0.146 / 0.159, **not measured**.
- Pushed 504863 (ppl 30.43) to the Hub: commit `535edc1bd3` (now the latest version).
- **Budget check (job 505032, W&B `8bw2x492`, plain `sbatch run.sbatch`):** 10334 steps, mean step 0.1455 s
  ((1569 s − 66 s eval) / 10334), saved at 1577 s, job total 1613 s (26:53), no deadline stop. Final eval
  **ppl 30.32** (loss 3.412), 917k tokens/s, MFU 39.9%. vs 504863 (10263 steps, 30.43): −0.11, about noise.
  Kept the 505032 model and deleted 504863's local copy (on the Hub). 505032 is not pushed.

## 2026-10-06 — NUMA binding (NCHC nano4: 2 × Xeon 8480+, 2 NUMA nodes)

**Question:** Does keeping CPUs and memory on the GPUs' NUMA node speed up training?

- Topology (job 505595): NUMA node 0 = CPUs 0–55, node 1 = 56–111. Our 2 GPUs (PCI 9A:00.0, DA:00.0, NVLink NV18)
  are both on **node 1**, but Slurm gives the job 24 CPUs split 12/12: `16-19,32-39` (node 0) + `72-75,88-95` (node 1),
  memory allowed on both nodes.
- Slurm here ignores `--gres-flags=enforce-binding` and `--sockets-per-node=1` (505605–505607: same CPU set), so the
  binding has to happen in the job: `numa_bind.sh` prints a `numactl` prefix for the first GPU's node, used by
  `run.sbatch` / `profile.sbatch` when `NUMA_BIND=local` (the job's CPUs on that node + its memory) or `NUMA_BIND=mem`
  (all 24 CPUs, local memory). Off by default.

Same-node A/B (505608, hgpn010, default setup, 100 steps, each mode twice, interleaved):

| Mode | CPUs visible | Median step (rep 1 / rep 2) | GPU busy (profile window) |
|---|---|---|---|
| off (default) | 24 (12 + 12) | 0.141 / 0.140 s | 100% / 98% |
| `local` | 12 (node 1) | 0.142 / 0.142 s | 100% / 102% |
| `mem` | 24 | 0.142 / 0.141 s | 104% / 97% |

- **No gain:** the GPU is already ~100% busy; host↔device traffic is just token ids, and the CPU-side work keeps up
  from either socket. `local` is ~1% slower (half the cores). Left off; the switch stays for future CPU-heavy setups.

## 2026-10-06 — Step time by batch size with compile (for the batch-size ramp)

| Global batch (per GPU) | Job | Median step | ms / sequence | Peak reserved |
|---|---|---|---|---|
| 64 (32) | 505613 | 0.081 s | 1.27 | 13.2 GiB |
| 128 (64) | 504833 | 0.140 s | 1.09 | 23.8 GiB |
| 256 (128, no accum) | 505614 | 0.264 s | **1.03** | 44.9 GiB |

- With compile, one 128-seq micro-batch is 6% faster per token than 64 × accum 2 (0.280 s); `--per_device_batch`
  default raised 64 → 128 (no change at global batch 128). `SEC_PER_STEP` 64 → 0.083, 256 → 0.272 (median × 1.03).
- Batch 64 is 16% slower per token, so time spent at batch 64 costs tokens.

## 2026-10-06 — Batch-size ramp (`--batch_ramp`)

**Question:** Does changing the global batch during training (small early, large late) beat a fixed 128?

`--batch_ramp "B1:f1,...,Bn"`: batch Bi for fraction fi of the training *time*; each stage's steps come from its own
`SEC_PER_STEP`, so every run has the same wall-clock budget. One micro-batch of B/2 per GPU (no grad accum) via a
seeded `RampBatchSampler` (accelerate's `BatchSamplerShard` accepts varying sizes); one static recompile per size.
LR (1.25e-3), warmup 1% and WSD 20% decay run over the total optimizer steps, unchanged. Smoke test 505631 OK
(983k tok/s in the 256 stage vs 920k at 128; recompiles cost a few seconds).

| Run | Job | W&B run | Stages (batch × steps) | Tokens | Final eval ppl | Δ ppl | Final eval loss |
|---|---|---|---|---|---|---|---|
| Fixed 128 (default) | 505032 | `8bw2x492` | 128 × 10334 | 1.35B | 30.32 | — | 3.412 |
| 64 → 128 (10% of time at 64) | 505635 | `80jdamxl` | 64 × 1817, 128 × 9300 | 1.34B | 31.24 | +0.92 | 3.442 |
| **128 → 256 (half / half)** | 505636 | `curujnbc` | 128 × 5167, 256 × 2773 | **1.40B** | **30.07** | **−0.25** | **3.404** |
| 64 → 128 → 256 | 505637 | `n4tnj8o4` | 64 × 1817, 128 × 5167, 256 × 2218 | 1.38B | 31.50 | +1.18 | 3.450 |
| Fixed 256 | 505638 | `3w1h3eb9` | 256 × 5547 | 1.45B | 30.76 | +0.44 | 3.426 |

All finished in 1590–1626 s with no deadline stop.

### Findings

- **New best: 128 → 256 at half time, ppl 30.07 (−0.25).** Batch 256 is too big early (fixed 256: +0.44), but in the
  second half it's fine and gives 6% more tokens/s (128/GPU micro-batch), so the run sees 4% more tokens.
- **Starting at batch 64 hurts (+0.9 to +1.2)**: batch 64 is 16% slower per token (fewer tokens overall), and it's behind
  from the first quarter on, at the same LR; with a smaller batch the gradient noise is higher, and 1.25e-3 may be
  too high for it. Not tuned further.
- −0.25 is about the size of single-seed noise seen before (up to ~0.4 mid-run), so it needs a confirming run. Next:
  switch point (30% / 70%) and a 128 → 256 → 512 ramp.
- Kept `~/gpt2_models/adamw-wu1-wsd20-compile-ramp128-256` (505636); deleted 505032, 505635, 505637, 505638.
  Pushed to the Hub: commit `2368f738d1` (latest version; sha256 `1a4d422b…` verified against the local file).

## 2026-10-06 — Document-level MinHash near-dedup (Jaccard ~0.8)

**Question:** Does removing near-duplicate documents from the training data (first 20 C4 shards, ~3.4B tokens)
improve C4 validation perplexity?

### Pipeline (`dedup_filter.py --stage dedup`, datatrove 0.10.1, `dedup_filter.sbatch dedup`)

- 5-grams of whitespace words of datatrove's normalized text (lowercased, punctuation removed), 20 bands × 12 hashes:
  P(caught) = 1 − (1 − J^12)^20 = 0.95 at J=0.85, 0.76 at 0.8, 0.24 at 0.7. One doc kept per cluster.
- datatrove's default spaCy word tokenizer made signatures ~17 docs/s per process (job 505612, would have taken ~9.5 h;
  `make_doc` was 57% of the time); whitespace tokens (as in Lee et al.) → ~9.7k docs/s total on 12 CPUs (~12 min).
- Validation untouched; `val.bin` copied from the raw data (byte-identical).
- Ops: job 505851 died between stages on `os.getcwd()` (network FS); `dedup_filter.sbatch` now `cd`s explicitly;
  the resubmit (505889) skipped the completed stages.

| | Raw (20 shards) | Deduped |
|---|---|---|
| Documents | 7,126,346 | 6,974,725 (−151,621, **−2.1%**) |
| Tokens | 3.409B | 3.369B (**−1.2%**) |

Duplicate clusters: 179,518 docs in clusters, largest 4,857 (templated pages: real-estate listings, dealer pages, SEO
image pages). A single shard alone had only 0.8% duplicates (smoke test 505522): most duplicates are across shards.

### Full run (current best setup: 128 → 256 batch ramp, equal time)

| Data | Job | W&B run | Final eval ppl | Final eval loss | Final train loss |
|---|---|---|---|---|---|
| Raw | 505636 | `curujnbc` | **30.07** | 3.404 | 3.418 |
| Deduped | 505942 | `0ugnnq52` | 30.48 (+0.41) | 3.417 | 3.444 |

Eval ppl at the same eval (raw / dedup): 4: 77.5 / 80.3; 14: 45.0 / 45.5; 23: 39.2 / 39.8; 33: 32.1 / 32.5; 38: 30.07 / 30.48.
A 1000-step sanity run first (505935 raw / 505938 dedup) trained cleanly (ppl 150.4 / 138.7, early-curve noise).

### Findings

- **Dedup is slightly worse on C4 validation: +0.41 ppl**, behind by 0.4–0.6 from the first quarter on (not just end
  noise), and its train loss is higher (3.444 vs 3.418): the removed near-duplicates are easy, templated text.
- This matches the Lee et al. effect: templates that recur in training also recur in validation, so a model that saw
  them (partly memorized) scores better on that part of validation. The near-dups are only 1.2% of tokens, so most
  of the gap likely comes from the validation docs that have templated near-twins in training.
- Not yet measured: perplexity split into validation docs with / without a near-duplicate in the training set.
- Kept the dedup model (505942) for that analysis; the raw-data best (505636) stays the best model.

## 2026-10-06 — Mild heuristic filtering on top of dedup

**Question:** Does dropping clearly unlearnable junk (extreme repetition, very short fragments, keyword stuffing),
mildly, improve C4 validation perplexity? (No quality classifier.)

### Filters (`dedup_filter.py --stage filter`, on the deduped data, job 505936, 1 h 21 min)

- Gopher repetition, defaults (line / paragraph / n-gram repetition; catches keyword stuffing and SEO templates).
- Gopher quality, loosened: < 25 words (spaCy tokens, punctuation counted), < 2 stop words, avg word length outside 3–10,
  symbol / bullet / ellipsis ratios. Dropped Gopher's alpha-word-ratio rule and its 50-word minimum: on one shard the
  defaults removed 22% (alpha 7.7%: abstracts and blog posts, because spaCy counts punctuation as words; < 50 words 7.3%).

| Step | Docs in | Dropped | Main reasons |
|---|---|---|---|
| Gopher repetition | 6,974,725 | 461,072 (6.6%) | dup 5-grams 155k, top 4-gram 134k, top 3-gram 81k, top 2-gram 41k |
| Gopher quality | 6,513,653 | 142,868 (2.2%) | < 2 stop words 92k, < 25 words 39k, long words 6k, bullets 4k |
| **Total** | | **603,940 (8.7%)** | tokens 3.369B → **3.167B (−6.0%; −7.1% vs raw)** |

### Full run (best setup: 128 → 256 batch ramp, equal time)

| Data | Job | W&B run | Final eval ppl | Final eval loss | Final train loss |
|---|---|---|---|---|---|
| Raw | 505636 | `curujnbc` | **30.07** | 3.404 | 3.418 |
| Deduped | 505942 | `0ugnnq52` | 30.48 (+0.41) | 3.417 | 3.444 |
| **Deduped + filtered** | 506330 | `obh7sr5b` | 30.95 (+0.88) | 3.432 | 3.490 |

Eval ppl at the same eval (raw / dedup / filtered): 4: 77.5 / 80.3 / 79.8; 9: 52.1 / 53.0 / 53.4; 18: 41.7 / 42.4 / 42.8;
28: 34.9 / 35.3 / 35.8; 38: 30.07 / 30.48 / 30.95. Sanity run (1000 steps, 506297) trained cleanly: ppl 112.9.

### Findings

- **Filtering adds another +0.47 ppl over dedup (+0.88 over raw)**, behind at every eval of the run. The 1000-step
  sanity runs (raw 150.4 / dedup 138.7 / filtered 112.9) pointed the other way; they're too short and noisy to rank
  data (different schedule, decay at step 800).
- Both cleaning steps move the training distribution away from C4 validation, which is unfiltered C4 web text:
  templated, repetitive and keyword-heavy pages are part of what the model is scored on. C4 is already cleaned
  (C4's line / page rules and 3-sentence exact dedup), so the extra cleaning mostly removes in-distribution text.
- The higher train loss (3.49 vs 3.42) shows the removed docs were easy (repetitive) text.
- **Conclusion for a C4-perplexity objective: train on raw C4.** Cleaning might still help downstream quality (not measured).
- Deleted the filtered model; raw 505636 stays the best (on the Hub). Kept the dedup model (505942) in case of the
  validation-overlap analysis.
