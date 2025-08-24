# Lieball-MT-PersRibsToOpenSimModel
# Patient-Specific Rib Integration for Adolescent Idiopathic Scoliosis OpenSim Models

This repository contains the complete workflow for integrating patient-specific rib geometries from adolescent Idiopathic Scoliosis patients into existing OpenSim musculoskeletal models. The process involves medical image segmentation, 3D reconstruction of ribs, geometric processing, model integration, and validation.

## Overview

The workflow transforms generic OpenSim musculoskeletal models into patient-specific models by:
- Extracting patient-specific rib centerlines from EOS biplanar radiographs
- Generating personalized 3D rib meshes
- Adapting muscle attachment points to maintain anatomical relationships
- Verifying the transformation 

## Dependencies

### Software Requirements
- MATLAB R2024b or later
- OpenSim GUI 4.5
- 3D Slicer
- Python 3.x

### MATLAB Packages
- `tlsm` package (from BFH GitLab)
- `mat2os` package (from BFH GitLab)
- OpenSim API for MATLAB

### Python Libraries
```bash
pip install numpy scipy vtk xml matplotlib
```

## Repository Structure

```
├── RibGeneratorsPers/
│   ├── preprocessing_moveRibCenterlines.m
│   ├── createRibBodiesFromTranslatedEOSCenterlines.m
│   └── rib_midline_extractor.py (modified)
│   └── translated_centerlines_NewSegmentation.txt
│   └── rib_centerlines_finalNewSegmentation.txt
├── GenericCenterline/
│   ├── extract_genericCenterline_fromVTP.py
│   └── generic_rib_centerlinesFromVTP/
├── RibPersonalisationImplementation/
│   └── rib_personalizer_functions/
│   └── PersonaliseRibsAndMusclePP.m
└── README.md
```

## Workflow

### Step 1: Base Model Creation
```matlab
% Create base MSK model with patient-specific spinal curvature using tlsm and  mat2os repositories from BFH GitLab
run via SGM02_AisPipeline.m of devBranchPhilippe from BFH GitLab
```
**Output**: Base model `SGM02.osim` and `geometry/` folder with generic rib files

### Step 2: Generic Rib Centerline Extraction

#### Method A: From .obj files (Medial Axis Approximation)
```python
python extract_genericCenterline_fromOBJ.py
```

#### Method B: From .vtp files (XML-based - Recommended)
```python
python extract_genericCenterline_fromVTP.py
```

### Step 3: Patient-Specific Centerline Extraction

**Important**: This step requires Tim Schär's repository.

1. Clone Tim's 3D-Ribcage-Reconstructor:  https://github.com/schaertim/3D-Ribcage-Reconstructor
2. Create predicted segmentations of EOS images
3. Manually refine provided segmentation predictions in 3D Slicer
4. 2. Place binaly labelmap files of the segmentations in appropriate folders
5. Use the modified `rib_midline_extractor.py` (includes added export function)
6. Add the following lines to the main script
   ```python
   p# Export centerlines
    centerlineFileName = "rib_centerlines_finalNewSegmentation.txt";
    extractorMidline.export_3d_midlines_as_txt(centerlineFileName)
   ```
8. Run the main script:
   ```python
   python main.py
   ```

**Output**: Patient centerline data with format:
```
rib_index,side,point_index,x,y,z
```
Where:
- `rib_index`: 1-12 (anatomical rib number)
- `side`: R/L (right/left) 
- `point_index`: 0-100 (posterior to anterior)
- `x,y,z`: 3D coordinates

### Step 4: Centerline Preprocessing
```matlab
preprocessing_moveRibCenterlines.m
```
- Translates centerlines to local coordinate system (origin at 0,0,0)
- Includes quality verification and visual inspection plots
- **Output**: Translated centerline data txt file 

### Step 5: 3D Rib Mesh Generation
```matlab
createRibBodiesFromTranslatedEOSCenterlines.m
```
**Process**:
- Fits cubic splines to centerlines
- Constructs local coordinate frames (tangent, normal, binormal vectors)
- Defines elliptical cross-sections
- Assembles triangular mesh surfaces

**Parameters to configure**:
- `n_spline_points`: Number of spline points
- `n_cross_points`: Circumferential points per cross-section
- `rib_width`, `rib_height`: Cross-section dimensions
- translated centerline file name if changed

**Output**: Personalized rib meshes named `[rib_number][R/L]_pers.obj`

### Step 6: Model Integration and Muscle Adaptation
```matlab
PersonaliseRibsAndMusclePP.m
```

**Main function**: `transform_rib_pathpoints`

**Process**:
1. Extract muscle path points attached to ribs
2. Compute transformations from generic to personalized geometry  
3. Calculate new muscle positions using relative position method:
   ```
   P_pers = C_pers + v
   ```
   Where:
   - `P_pers`: New muscle attachment point
   - `C_pers`: Corresponding point on personalized centerline
   - `v`: Offset vector from generic centerline

4. Update model with personalized muscle attachments
5. Replace generic rib meshes with patient-specific meshes
6. Generate verification plots

**Output**: 
- Intermediate model (updated muscle points only)
- Final personalized model (updated muscle points + personalized meshes)

### Step 7: Model Validation
Perform static optimization and joint reaction analysis in OpenSim to compare:
- Muscle activation patterns between base and personalized models
- Joint reaction forces during neutral standing posture
- Analysis over 1-second duration (11 time points)

## Key Features

### Transformation Algorithm
- **Relative Position Preservation**: Maintains spatial relationships between muscles and ribs
- **Centerline-Based Mapping**: Uses rib centerlines as geometric references
- **Affine Transformations**: Maps between generic and patient-specific geometries

### Quality Assurance
- Visual verification plots for all transformation steps
- Numerical verification (tolerance: 10⁻¹⁰)
- Mesh integrity checks (connectivity, vertex counts)
- Muscle path length change tracking

### File Naming Convention
- Generic ribs: `[rib_number][R/L].[extension]`
- Personalized ribs: `[rib_number][R/L]_pers.[extension]`
- Example: `4R_pers.obj` (right rib 4, personalized)

## Usage Example

Complete workflow execution:
```matlab
% 1. Create base model
SGM02_AisPipeline.m

% 2. Process centerlines (after EOS segmentation step)
preprocessing_moveRibCenterlines.m

% 3. Generate personalized meshes
createRibBodiesFromTranslatedEOSCenterlines.m

% 4. Integrate into final model
PersonaliseRibsAndMusclePP.m
```

## Patient Data

**Case Study**: SGM02
- 13-year-old female with adolescent idiopathic scoliosis
- Double curve pattern: T6-T11 (21.8°), T11-L3 (22.7°)
- Height: 1.65m, Weight: 45kg
- Treatment: Physiotherapy and bracing



## Citation and Acknowledgments

**Base Components**:
- MSK model adapted from Schmid et al. (2020)
- Spine alignment from Cedric Rauber (2022) 
- **EOS segmentation and 3D reconstruction**: Tim Schär (2025) https://github.com/schaertim/3D-Ribcage-Reconstructor
**Key Modification**: Added `export_3d_midlines_as_txt()` function to `rib_midline_extractor.py` for centerline data export.


## Contact

For any questions, feel free to reach out

