# DeSARD Synthetic Blender Generator

This folder contains the Blender scene and Python scripts used to generate the synthetic subset of DeSARD, a desert SAR dataset for aerial human detection. The generator renders top-down desert scenes with randomized dunes, vegetation, lighting, camera placement, and human targets, then exports both images and YOLO-format labels.

## Project contents

| File / folder | Purpose |
|---|---|
| `scene.blend` | Main Blender scene used for synthetic image generation. |
| `driver.py` | Embedded Blender text script that randomizes the scene, renders images, and writes YOLO labels. |
| `organize_synthetic.py` | Utility script that consolidates rendered height folders into `images/`, `labels/`, and `metadata.csv`. |
| `Shrub.blend` | Shrub vegetation asset. |
| `my grass.blend` | Grass vegetation asset. |
| `tree/tree1.blend` | Tree vegetation asset. |

## What `driver.py` does

`driver.py` is the main generation script. It is stored as a Blender text block inside `scene.blend`.

The script performs four main steps:

1. Randomizes the desert scene.
2. Randomizes the camera position, altitude, and rotation.
3. Renders an image.
4. Computes and writes a YOLO label file.

The main classes/functions are:

| Component | Role |
|---|---|
| `Interpolate` | Small helper for mapping values across ranges. |
| `SceneController` | Controls dunes, vegetation, time of day, human visibility, human pose, rotation, and shirt color. |
| `CameraController` | Controls the selected camera height, position, rotation, and valid sampling bounds. |
| `human_pixel_bbox()` | Projects the rendered human mesh into image space and returns a pixel bounding box. |
| `write_yolo_label()` | Writes one YOLO label line, or an empty file for negative samples. |
| `draw_bbox_debug()` | Creates debug copies with the projected bounding box drawn on top. |
| `DesertPhoto` | Coordinates randomization, rendering, label creation, and reset. |

## Randomized parameters

The generator randomizes several scene and camera properties, including:

- Dune shape seed
- Dune aggressiveness
- Grass spawn seed and density
- Shrub spawn seed and density
- Tree spawn seed and density
- Time of day / HDRI rotation
- Human presence or absence
- Human pose
- Human rotation
- Human shirt color
- Camera x/y position
- Camera altitude
- Camera rotation

## Human labels

The project uses YOLO-format object detection labels. Each image has a matching `.txt` file.

For human images, the label format is:

```txt
0 cx cy w h
```

where:

- `0` is the class ID for `human`.
- `cx` and `cy` are the normalized bounding-box center coordinates.
- `w` and `h` are the normalized bounding-box width and height.

For images without a human, the script writes an empty `.txt` file. This is a valid YOLO negative sample.

## Visibility filtering

The script filters out human images where the projected body is not sufficiently visible.

Important constants:

```python
MIN_VISIBLE_FRACTION = 0.7
EDGE_MARGIN_PX = 2
```

Meaning:

- `MIN_VISIBLE_FRACTION = 0.7` keeps a human image only if at least 70% of the projected box is inside the frame.
- `--min-visible 1.0` makes the filter strict, requiring the full human projection to be inside the frame.
- `EDGE_MARGIN_PX = 2` gives a small edge clearance in strict mode.

Images that fail the visibility check are deleted and are not kept in the dataset.

## Running the generator

Run the script from Blender in background mode:

```bash
blender -b scene.blend --python-text driver.py -- --height 50 --count 100 --camera rpi
```

Generate images without humans:

```bash
blender -b scene.blend --python-text driver.py -- --height 50 --count 100 --camera rpi --no-human
```

Use the Arducam camera model:

```bash
blender -b scene.blend --python-text driver.py -- --height 50 --count 100 --camera arducam45
```

Generate debug bounding-box copies:

```bash
blender -b scene.blend --python-text driver.py -- --height 50 --count 20 --camera rpi --debug-bbox
```

Require the full human to be inside the frame:

```bash
blender -b scene.blend --python-text driver.py -- --height 50 --count 100 --camera rpi --min-visible 1.0
```

Available command-line arguments:

| Argument | Description |
|---|---|
| `--height` | Camera height / altitude. Default: `50`. |
| `--count` | Number of images to render. Default: `1000`. |
| `--no-human` | Render negative samples without a human. |
| `--debug-bbox` | Save debug images with bounding boxes drawn on them. |
| `--camera` | Camera choice. Options: `rpi`, `arducam45`. Default: `rpi`. |
| `--min-visible` | Override the minimum visible fraction. Use `1.0` for strict fully-in-frame labels. |

## Raw render output

By default, `driver.py` writes renders to:

```txt
~/Desktop/blender_out_low_res
```

The output is grouped by height and category:

```txt
blender_out_low_res/
  height_50.0/
    with human/
      sitting/
        image.jpg
        image.txt
      standing/
        image.jpg
        image.txt
    without human/
      image.jpg
      image.txt
```

Each rendered `.jpg` has a matching `.txt` YOLO label.

## Organizing the synthetic dataset

`organize_synthetic.py` consolidates rendered images and labels into a flat dataset layout.

Expected source structure:

```txt
dataset/synthetic/
  height_<altitude>/
    with human/
      sitting/
        *.jpg
        *.txt
      standing/
        *.jpg
        *.txt
    without human/
      *.jpg
      *.txt
```

Run:

```bash
python organize_synthetic.py --root dataset/synthetic --destination dataset/synthetic --force
```

Use `--copy` if hard links are not desired:

```bash
python organize_synthetic.py --root dataset/synthetic --destination dataset/synthetic --force --copy
```

The organizer creates:

```txt
dataset/synthetic/
  images/
    *.jpg
  labels/
    *.txt
  metadata.csv
```

The generated `metadata.csv` has the columns:

```txt
image, altitude, category
```

## Dependencies

The generator runs inside Blender and uses Python modules available in Blender plus NumPy. The organizer is a normal Python script and uses:

```bash
pip install pandas tqdm
```
