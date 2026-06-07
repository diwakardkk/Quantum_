# Dataset Folder

Put your dataset here if you want the default `config.yaml` to use it directly.

Supported structures:

```text
data/
  train/glioma/...
  val/glioma/...
  test/glioma/...
```

or:

```text
data/
  glioma/...
  meningioma/...
  pituitary/...
  no_tumor/...
```

The code also auto-detects nearby folders named `Brain_Tumor_MRI_Dataset` or `dataset`.
Your existing folder can stay outside this project; the scripts will try to find it.
