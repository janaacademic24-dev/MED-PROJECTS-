# MED-PROJECTS-
This repository is a growing collection of projects focused on the intersection of AI, medical imaging, and interactive visualization. Designed for students, researchers, and developers, these projects explore how deep learning and Python-based tools can be used to analyze, segment, and visualize anatomical structures in 2D and 3D.
# 3D Organ Segmentation and Visualization

This project visualizes three organs (Liver, Lungs , Brain) in 3D and performs automated segmentation using AI models in Python and Interactive controls that allow the viewer to change the colors,opacity and visbility

## steps 
1) Choosing 3 organs 
2) Gathering 3D medical data: using available database / ensure data formats (DICOM , NIFTI )
3) Using AI models 
4) Using metrics like (Dice Score / IoU (Jaccard) / Hausdorff Dist )
5) Using python codes for interactive plots and 3D rendering 
6) Dash or Streamlit for web-based apps ( for allowing changing visibility, transparency, colors)
7) Testing
8) Writing documentation, including images and videos. 


## Features

- 3D visualization using matplotlib/pyvista/plotly/skimage/Nibabel/scipy/Numpy
- Automated segmentation using trained deep learning models
- Organ-specific models
- User-friendly script execution
- 
## Evaluation metric 
LIVER ![Ai evaluation ](https://github.com/user-attachments/assets/74082080-e738-46a2-b89b-0bf9238da332)

Lung <img width="1702" height="747" alt="capture_251001_112050" src="https://github.com/user-attachments/assets/6f69e739-e66f-440e-aed9-ad350040567e" />

brain 

<img width="1888" height="1007" alt="capture_251001_114255" src="https://github.com/user-attachments/assets/9d09f725-464a-4177-9fc2-3f0b8e6488bf" />



## Screenshots

Liver ![liver 2](https://github.com/user-attachments/assets/f4ace21f-1a6b-47e3-b37f-d73e5054c9ed)

Lungs ![lungs segmentation](https://github.com/user-attachments/assets/ef1786d7-d8ed-4471-afa5-fd92f543e793)

Brain <img width="1672" height="648" alt="image" src="https://github.com/user-attachments/assets/3630a80a-4d4e-4ad0-a4fd-ba29bf47875a" />


## 3D Visualization 

### Liver
<img width="605" height="546" alt="image" src="https://github.com/user-attachments/assets/72c9ded2-8686-4096-bde6-0e13b8640c28" />


### Lungs
(<img width="964" height="654" alt="Screenshot 2025-09-29 162536" src="https://github.com/user-attachments/assets/a25920b0-f5d7-4765-b0bc-7eee2ae7a2a1" />

### brain
<img width="1007" height="704" alt="capture_251001_113511" src="https://github.com/user-attachments/assets/708a9377-38c8-415c-9d65-4cd7c55d2a74" />



## Installation

```bash
pip install -r requirements.txt
