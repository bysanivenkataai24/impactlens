# models/

Checkpoints are **not** committed to git (see `.gitignore`).

`impactlens export --run <run dir> --model-id <id>` writes a deployable artifact
directory here:

```
models/<model_id>/
    <model_id>.pt          checkpoint with embedded provenance
    model_manifest.json    identity, SHA-256, classes, operating point, metrics, licences
    SHA256SUMS
    service.env            MODEL_PATH / MODEL_ID / MODEL_REVISION / MODEL_SHA256 ...
```

For hosting, upload the `.pt` file to a Hugging Face model repository and
configure the service with `MODEL_REPO`, `MODEL_FILENAME` and a full
40-character `MODEL_REVISION` commit SHA, plus `MODEL_SHA256`. A file placed
here before `docker build` is copied into the image at `/app/models/` and can
be served with `MODEL_PATH` instead.

A `.pt` file is a Python pickle: load only checkpoints whose origin you trust,
and always pin `MODEL_SHA256`.
