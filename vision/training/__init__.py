"""
vision/training/
================
The Vision tab's *trained* component — a small classifier over frozen
BiomedCLIP features, learned from the MURA musculoskeletal X-ray set.

This is the "hybrid" from the design: BiomedCLIP (already used by the Vision
tab's correction memory) turns each image into a 512-d vector; a small head
learns to map those vectors to a bone label. At inference the head's prediction
is fed to the VLM (``llava:13b``) as grounding, so the conversational answer is
anchored to a model trained on real bone data rather than the VLM guessing cold.

Two stages, two scripts:

  mura_dataset    parse MURA image paths → (region, patient, normal/abnormal),
                  with a patient-disjoint train/val carve-out (no patient leaks
                  across splits — the #1 way to fake a good accuracy).
  cache_features  run every image through BiomedCLIP once and cache the vectors
                  (resumable; ``--limit`` for a quick smoke run). GPU-bound —
                  this is the step worth running on the Jetson.
  train_head      train the classifier head on the cached vectors and report
                  per-region accuracy on MURA's held-out valid split. Trivial
                  (seconds); runs anywhere once the features are cached.

Feature extraction is the only expensive part, and it is inference, not
training — which is why the Jetson (built for inference) handles it fine even
though it is the deployment box.
"""
