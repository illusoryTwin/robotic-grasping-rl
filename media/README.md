# Media Assets

This directory contains visual documentation for the project.

## Structure

```
media/
├── screenshots/          # Simulation screenshots and visualizations
│   ├── scene_overview.png
│   ├── object_spawning.png
│   ├── successful_grasp.png
│   └── training_progress.png
└── videos/              # Training/evaluation videos (optional)
    └── (links to external storage)
```

## Screenshot Naming Convention

Use descriptive names with snake_case:
- `scene_overview.png` - Full scene with all objects
- `object_spawning.png` - Initial object placement
- `successful_grasp_tincan.png` - Successful grasp of specific object
- `failure_mode_<type>.png` - Examples of failure modes
- `training_curve_<metric>.png` - Training curves from tensorboard

## Recommended Image Specifications

- **Format**: PNG (for screenshots), JPG (for photos)
- **Resolution**: 1920×1080 or higher for main figures
- **Size**: <2MB per image (compress if needed)
