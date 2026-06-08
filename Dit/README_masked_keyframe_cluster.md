# Masked VI/IR Keyframe Clustering

'masked_keyframe_cluster.py' is used to process infrared/visible dual-mode video sequences:

1. Read the 'vi/', 'ir/', and 'init.txt' of each sequence in the root directory of the dataset.
2. Read each frame of mask in the same name sequence folder in the root directory of the mask.
3. For visible light, load and multiply the mask by RGB three channels; for infrared, load and multiply the mask by a single grayscale channel.
4. Perform MSE-based keyframe extraction for each sequence, fixing the first frame as the keyframe.
5. Perform KMeans clustering for visible light keyframe collections and infrared keyframe collections, with the default cluster count being the number of video sequences.
6. Save the cluster centers of the two modalities as images, i.e., common feature maps.

## Enter directory format

```text
dataset_root/
  seq_001/
    vi/
      000001.jpg
      000002.jpg
    ir/
      000001.png
      000002.png
    init.txt
  seq_002/
    vi/
    ir/
    init.txt

mask_root/
  seq_001/
    000001.png
    000002.png
  seq_002/
    000001.png
    000002.png
```

'init.txt' per line format:

```text
Top left x, top left y w h
```

The current script reads and checks the number of 'init.txt' lines, but keyframe extraction and clustering use the entire mask image. If you want to handle only the target box area later, you can add bbox clipping logic after loading the mask.

## Running Example

```powershell
python .\masked_keyframe_cluster.py `
  --dataset-root D:\data\dataset `
  --mask-root D:\data\masks `
  --output-dir D:\data\common_features `
  --threshold 0.5 `
  --feature-size 224x224
```

If the mask is a soft mask, it retains grayscale weights by default; If you want to treat it as a binary mask:

```powershell
python .\masked_keyframe_cluster.py `
  --dataset-root D:\data\dataset `
  --mask-root D:\data\masks `
  --output-dir D:\data\common_features `
  --binary-mask
```

If you want to save the selected mask keyframes at the same time:

```powershell
python .\masked_keyframe_cluster.py `
  --dataset-root D:\data\dataset `
  --mask-root D:\data\masks `
  --output-dir D:\data\common_features `
  --save-keyframes
```

## Output the directory

```text
output_dir/
  vi_common_features/
    cluster_000.png
    cluster_001.png
    clusters.json
  ir_common_features/
    cluster_000.png
    cluster_001.png
    clusters.json
  run_metadata.json
```

Among them:

- 'vi_common_features/': Visible light common feature map.
- 'ir_common_features/': Infrared common feature map.
- 'clusters.json': Keyframe member information corresponding to each cluster center.
- 'run_metadata.json': Runtime parameters, number of keyframes, and alarm information.

## Dependence

```powershell
pip install numpy pillow
```