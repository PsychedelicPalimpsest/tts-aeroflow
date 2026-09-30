# Train new AeroFlow voices on Kaggle

This guide covers **two independent models**: LJSpeech and HiFi speaker **9017**.
Start each from random weights, use its complete corpus, and keep its checkpoints
in separate directories. You do not need the previously trained Downloads files.

The implemented workflow has two stages:

| Stage | Command | What learns | Starting weights |
|---|---|---|---|
| A: Joint TTS training | `scripts/train_kaggle.py` | Text, durations, acoustic encoder, flow, decoder | Random for a new model |
| B: Acoustic decoder training | `scripts/train_vocoder.py` | Decoder and training-only discriminators | Stage A checkpoint for that voice |

Stage A still uses the legacy spectral reconstruction objective alongside text
and flow losses. Stage B addresses its suspected contribution to robotic timbre
using mel, adversarial and feature-matching losses. It freezes the encoder and
flow so their learned latent representation stays compatible. The vocoder trainer
cannot train a complete TTS model from random weights on its own.

The padding defect has been fixed in both paths. These changes have passed local
CPU tests and short real-audio training checks; full-corpus GPU convergence and
perceptual improvement have not yet been established. Epoch/step budgets below
are starting points, not guarantees of natural speech. See
[VOCODER_REPAIR.md](VOCODER_REPAIR.md) for the objective and diagnostic results.

## 1. Prepare the notebook and storage

Enable a GPU accelerator. Enable Internet if loading from Hugging Face. Check
actual GPU availability, free disk space and the session limit shown by Kaggle;
do not assume a fixed hardware allocation, storage quota or runtime allowance.

Use **one GPU per training process** for this guide. Stage B has no DDP support;
do not launch it with `torchrun`. Stage A contains a distributed path, but its
multi-GPU behavior was not validated by the local checks. The commands below
select GPU 0 explicitly and work without relying on that path.

In a notebook Python cell:

```python
import os
import shutil
import torch

assert torch.cuda.is_available(), "Enable a GPU accelerator before training"
print(torch.__version__)
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

# Set BEFORE importing datasets/aeroflow or starting either trainer.
# These are cache locations, not a promise of additional disk capacity.
os.makedirs('/kaggle/tmp', exist_ok=True)
os.environ['HF_HOME'] = '/kaggle/tmp/hf_home'
os.environ['HF_HUB_CACHE'] = '/kaggle/tmp/hub'
os.environ['HF_DATASETS_CACHE'] = '/kaggle/tmp/hf_cache'
for path in ['/kaggle/tmp', '/kaggle/working']:
    print(path, 'free GiB:', round(shutil.disk_usage(path).free / 2**30, 1))
```

In a fresh notebook session, clone the repository and install dependencies:

```python
!git clone https://github.com/PsychedelicPalimpsest/tts-aeroflow.git /kaggle/working/ttx
%cd /kaggle/working/ttx
!python -m pip install -q numpy scipy soundfile "datasets[audio]" pytest
!python scripts/train_kaggle.py --help
!python scripts/train_vocoder.py --help
```

Keep the notebook's CUDA-compatible PyTorch installation. Record the repository
commit and package versions with the run. On subsequent sessions, restore or
clone the same code revision before resuming. All remaining shell examples can
be pasted into notebook cells beginning with `%%bash`.

Put checkpoints under `/kaggle/working` and preserve notebook outputs before the
session ends. Caches under `/kaggle/tmp` are disposable. Distinct paths do not
necessarily live on distinct storage devices. Downloading and preparing a full
HF split can require space for both source shards and prepared Arrow data.
Filtering speaker 9017 happens after loading the split and does not guarantee
that only that speaker's audio will be downloaded.

## 2. Supply complete datasets

### LJSpeech

Attach the complete LJSpeech dataset as a Kaggle input. Replace the example path
`/kaggle/input/ljspeech-1-1/LJSpeech-1.1` below with the directory containing
`metadata.csv`:

```text
LJSpeech-1.1/
  metadata.csv
  wavs/                 # audio/ is also supported
    LJ001-0001.wav
    LJ001-0002.wav
    ...
```

Both trainers use the normalized transcript column, mix to mono, resample to
24 kHz with an anti-aliasing filter and normalize peak level to 0.95. They filter
out recordings outside their supported duration range. Stage A defaults to
0.5–12 seconds for LJ; Stage B also excludes clips shorter than its crop length.
If using Stage A's alternate metadata/audio-directory flags, arrange the same
corpus in this standard layout for Stage B, whose CLI takes `--data-root`.

Check the number of actual loadable recordings, not just metadata rows:

```python
from aeroflow.dataset.ljspeech import LJSpeechDataset
lj = LJSpeechDataset('/kaggle/input/ljspeech-1-1/LJSpeech-1.1')
print('Loadable LJ recordings after filtering:', len(lj))
```

The local `/tmp/lj` used during development contained **five WAVs**, despite its
13,100-row metadata file. It was suitable for diagnostics only. Do not use that
small copy for a full training run.

### HiFi speaker 9017

Use the full [MikhailT/hifi-tts](https://huggingface.co/datasets/MikhailT/hifi-tts)
repository, `clean` configuration, `train` split, and speaker `9017` throughout.
The local checks used `MikhailT/hifi-tts-light`; its clean training split has only
three clips for this speaker, so it is unsuitable for training a new voice.

Stage A supports both cached map access (`--dataset-source hf`) and streaming
(`--dataset-source hf-streaming`). Stage B currently requires cached map access;
it does not support streaming or a local HiFi manifest. Budget disk space for
that second stage before starting. Native HiFi audio is converted to 24 kHz by
the adapter. The stage A examples cap clips at 10 seconds; Stage B caps at 12.

For cached access, this reads metadata and reports the filtered count, but may
first download and prepare the full requested split:

```python
from aeroflow.dataset.hf_hifi_tts import HuggingFaceHiFiTTSDataset
hifi = HuggingFaceHiFiTTSDataset(
    repo_id='MikhailT/hifi-tts', subset='clean', split='train',
    speaker_ids=('9017',), cache_dir='/kaggle/tmp/hf_cache',
    max_duration_s=10.0,
)
print('Loadable HiFi 9017 recordings:', len(hifi))
```

## 3. Stage A: train each model from scratch

Use `--no-auto-resume`, a new output directory, and **no** `--resume-path` or
`--finetune`. Merely changing the output directory is insufficient if automatic
resume is enabled: it searches attached input datasets and other checkpoint
folders too. `--finetune` starts from existing weights and is not scratch training.
An explicitly requested checkpoint that does not exist now fails immediately.

### New LJSpeech model

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_kaggle.py \
  --dataset-source ljspeech \
  --ljspeech-root /kaggle/input/ljspeech-1-1/LJSpeech-1.1 \
  --checkpoint-dir /kaggle/working/lj_joint \
  --no-auto-resume \
  --batch-size 8 --num-workers 2 --epochs 100 --lr 2e-4 \
  --save-interval-steps 500 --sample-interval-steps 1000 \
  --max-hours 10
```

### New HiFi 9017 model

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_kaggle.py \
  --dataset-source hf --hf-repo-id MikhailT/hifi-tts \
  --hf-subset clean --hf-split train --hf-speaker 9017 \
  --hf-cache-dir /kaggle/tmp/hf_cache \
  --checkpoint-dir /kaggle/working/hifi9017_joint \
  --no-auto-resume \
  --batch-size 8 --num-workers 2 --epochs 100 --lr 2e-4 \
  --save-interval-steps 500 --sample-interval-steps 1000 \
  --max-hours 10
```

Choose a `--max-hours` value shorter than your actual session allowance, allowing
for dataset setup and output preservation. The stage A watchdog starts after
setup, so download time is not included. Batch size 8 is an initial setting, not
a measured memory guarantee; reduce it if needed. Keep one voice per run.

For streaming HiFi, replace `--dataset-source hf` with `hf-streaming` and add
`--steps-per-epoch 1000 --hf-shuffle-buffer 1000`. A streaming epoch then means a
configured number of batches, not one complete corpus pass. Track update counts
and data exposure when comparing runs. This does not enable streaming in Stage B.

Stage A writes `checkpoint_latest.pt`, `checkpoint_best.pt` and listening samples.
Its “best” is selected from a single training batch's combined loss, not held-out
perceptual quality. Keep checkpoints with their fixed listening samples. Before
Stage B, require intelligible, stable text output and recognizable reconstruction
from recordings. If text alignment or pronunciation is broken, decoder-only
training cannot repair it. A robotic texture persisting after otherwise stable
training is a reason to evaluate Stage B rather than blindly extend Stage A.

## 4. Resume Stage A across sessions

Save the previous run's outputs and attach them as a Kaggle input. Use an
**explicit** matching checkpoint, repeat the same dataset and training settings,
and write new outputs under `/kaggle/working`. For example:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_kaggle.py \
  --dataset-source ljspeech \
  --ljspeech-root /kaggle/input/ljspeech-1-1/LJSpeech-1.1 \
  --resume-path /kaggle/input/lj-joint-run/lj_joint/checkpoint_latest.pt \
  --checkpoint-dir /kaggle/working/lj_joint \
  --no-auto-resume \
  --batch-size 8 --num-workers 2 --epochs 100 --lr 2e-4 \
  --save-interval-steps 500 --sample-interval-steps 1000 \
  --max-hours 10
```

For HiFi, use the dataset/cache flags from its fresh-run command and the saved
`hifi9017_joint/checkpoint_latest.pt`. `--epochs` is the total target, not an
additional number of epochs. The trainer restores optimizer, scaler, scheduler,
step, epoch and RNG; it does not save an exact mid-epoch batch cursor, so some data
may repeat when resuming an interrupted epoch.

If extending training beyond the original epoch budget, choose a new total
`--epochs` and use `--reset-lr --lr <chosen-rate>` to reinitialize the cosine
schedule for the remaining epochs. This option also clears optimizer momentum.
Simply changing `--epochs` while restoring the old scheduler does not extend its
saved cosine period. Avoid `--finetune` when continuing the same run: it resets
step and epoch bookkeeping as well as optimizer state.

## 5. Stage B: train the acoustic decoder for each new model

Start from that voice's Stage A checkpoint, using a separate output directory.
There is no need to load any older pretrained voice. Select a checkpoint after
reviewing its samples; the examples use the latest one.

### LJSpeech decoder

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_vocoder.py \
  --dataset lj --data-root /kaggle/input/ljspeech-1-1/LJSpeech-1.1 \
  --checkpoint /kaggle/working/lj_joint/checkpoint_latest.pt \
  --output /kaggle/working/lj_vocoder --device cuda \
  --steps 100000 --batch-size 4 --workers 2 --threads 2 \
  --save-every 250 --validate-every 1000 --validation-items 16
```

### HiFi 9017 decoder

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_vocoder.py \
  --dataset hifi --hf-repo MikhailT/hifi-tts --hf-split train --speaker 9017 \
  --cache-dir /kaggle/tmp/hf_cache \
  --checkpoint /kaggle/working/hifi9017_joint/checkpoint_latest.pt \
  --output /kaggle/working/hifi9017_vocoder --device cuda \
  --steps 100000 --batch-size 4 --workers 2 --threads 2 \
  --save-every 250 --validate-every 1000 --validation-items 16
```

If starting Stage B in a later notebook session, replace `--checkpoint` with the
Stage A file under its attached `/kaggle/input/...` location. Note the different
flag names: Stage A uses `--hf-repo-id`/`--hf-speaker`/`--hf-cache-dir`; Stage B
uses `--hf-repo`/`--speaker`/`--cache-dir`.

Stage B uses more memory for its training-only critics. Start with batch size 4
and adjust to measured memory. The full utterance is reconstructed before valid
0.68-second waveform crops are selected for losses; a short crop does not cap the
memory needed for the complete decoder pass.

Stage B **does not have a wall-clock watchdog**. Set a session-sized total step
target from observed throughput, use frequent saves, and preserve outputs before
the session expires. The 100,000-step example is a total experimental budget
across sessions, not a promise that one notebook session will finish it. An
interrupted run can resume from its last completed atomic save.

## 6. Resume Stage B and choose a final model

Resume from **Stage B's** `checkpoint_latest.pt`, repeating its original
configuration. For LJ:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_vocoder.py \
  --dataset lj --data-root /kaggle/input/ljspeech-1-1/LJSpeech-1.1 \
  --resume /kaggle/input/lj-vocoder-run/lj_vocoder/checkpoint_latest.pt \
  --output /kaggle/working/lj_vocoder --device cuda \
  --steps 100000 --batch-size 4 --workers 2 --threads 2 \
  --save-every 250 --validate-every 1000 --validation-items 16
```

For HiFi, use the dataset/cache flags from its Stage B command and its own
`hifi9017_vocoder/checkpoint_latest.pt`. `--steps` counts total decoder updates,
including previous sessions. Do not supply `--checkpoint` together with `--resume`.
Resume validates the configuration and dataset filename order; keep data paths,
filtering and seed consistent. It restores both optimizers, discriminators,
scalers and RNG, with step-derived sampling and crops.

| File | Purpose |
|---|---|
| Stage A `checkpoint_latest.pt` | Resume Stage A or initialize Stage B |
| Stage B `checkpoint_latest.pt` | Resume Stage B; includes training critics/optimizers |
| Stage B `model_latest.pt` | Latest evaluated model for inference |
| Stage B `model_best_mel.pt` | Lowest validation mel loss, potentially the untouched baseline |
| Stage B `split.json`, `config.json`, `train.jsonl` | Data split, run settings, loss history |
| Stage B `baseline/`, `validation_*/` | Metrics, original/reconstructed audio and fixed-seed text samples |

Keep the baseline and earlier best-mel exports when attaching runs in later
sessions; the resume command restores training state but does not copy older
listening folders or exports into the new output directory. Do not resume the
legacy Stage A objective after finishing Stage B: it can undo decoder training.
Inference exports lack the optimizer state expected by Stage A.

Listen to original/reconstruction pairs and fixed text samples at each validation.
Magnitude metrics are not perceptual scores. Stage B reserves validation clips
from its own updates; those recordings may have been used by Stage A. For a truly
unseen final evaluation, reserve recordings outside both stages. The current
Stage A trainer has no built-in held-out validation split.

Synthesize with an explicitly selected final checkpoint:

```bash
python -m aeroflow 'The sound of the wind was soft and low.' \
  --checkpoint /kaggle/working/lj_vocoder/model_latest.pt \
  --device cuda -o /kaggle/working/lj_sample.wav

python -m aeroflow 'The sound of the wind was soft and low.' \
  --checkpoint /kaggle/working/hifi9017_vocoder/model_latest.pt \
  --device cuda -o /kaggle/working/hifi9017_sample.wav
```

Compare the raw model output without VoiceFixer or phase refinement first. An
optional restoration pass should not hide whether the decoder training helped.

## Optional pronunciation audit

If pronunciations are a separate problem, run the audit before Stage A and
review a small sample before applying it to the corpus:

```bash
python -m pip install cmudict transformers faster-whisper
python scripts/audit_pronunciations.py \
  --source hf-streaming --hf-repo-id MikhailT/hifi-tts --hf-speaker 9017 \
  --limit 200 --output /kaggle/working/pronunciation_sample.jsonl \
  --lexicon-output /kaggle/working/pronunciation_lexicon.json
```

For LJ, replace the source flags with `--source ljspeech --ljspeech-root <root>`.
After inspecting decisions, omit `--limit` for a full audit. Preserve its output
and add `--pronunciation-manifest <file.jsonl>` to Stage A with the same corpus,
speaker and split. For larger audits, `--new-items` and `--resume-from` allow
continuation. See [PRONUNCIATION_AUDIT_REAL_DATA.md](PRONUNCIATION_AUDIT_REAL_DATA.md)
and [PRONUNCIATION_AUDIT_LJSPEECH.md](PRONUNCIATION_AUDIT_LJSPEECH.md) for results
and limitations. The decoder trainer does not consume the audit manifest.

## Optional implementation checks

Before a long GPU run, the local behavioral suite can be run with:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest tests -q -k 'not integration'
```

For a short decoder plumbing check, use a Stage A checkpoint with complete
matching data and lower `--steps`, `--batch-size` and `--validation-items` in a
separate output directory. Small-critic or tiny-dataset diagnostic checkpoints
are not production voices. Full GPU/AMP validation remains necessary on the
actual training hardware.
