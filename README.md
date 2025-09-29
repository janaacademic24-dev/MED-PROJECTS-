# MED-PROJECTS-
This repository is a growing collection of projects focused on the intersection of AI, medical imaging, and interactive visualization. Designed for students, researchers, and developers, these projects explore how deep learning and Python-based tools can be used to analyze, segment, and visualize anatomical structures in 2D and 3D.
# 3D Organ Segmentation and Visualization

This project visualizes three organs (Liver, Lungs , Pancreas) in 3D and performs automated segmentation using AI models in Python.
## steps 
1) Choosing 3 organs 
2) Gathering 3D medical data: using available database / ensure data formats (DICOM , NIFTI )
3) Using AI models 
4) Using metrics like (Dice Score / IoU (Jaccard) / Hausdorff Dist )
5) Using python codes for interactive plots and 3D rendering 
6) Dash or Streamlit for web-based apps / Dash or Streamlit for web-based apps ( for allowing changing visibility, transparency, colors)
7) Testing
8) Writing documentation, including images and videos. 

## Dataset used
andrewmvd/liver-tumor-segmentation

https://radiopaedia.org/cases/covid-19-pneumonia-85


https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar

## Features

- 3D visualization using matplotlib/pyvista/plotly/skimage/Nibabel/scipy/Numpy
- Automated segmentation using trained deep learning models
- Organ-specific models
- User-friendly script execution
## Evaluation metric 
![Ai evaluation ](https://github.com/user-attachments/assets/74082080-e738-46a2-b89b-0bf9238da332)




## Screenshots
Pancreas ![pancreas segmentation ](https://github.com/user-attachments/assets/02488694-64bf-4e96-b14a-7cf60847f156)

liver ![liver 2](https://github.com/user-attachments/assets/f4ace21f-1a6b-47e3-b37f-d73e5054c9ed)

lungs ![lungs segmentation](https://github.com/user-attachments/assets/ef1786d7-d8ed-4471-afa5-fd92f543e793)


### Liver
![Liver  3D](![liver 1](https://github.com/user-attachments/assets/afb60cd9-bffa-4f96-a5c2-6c65b4e2bf45)
)
)

### Lungs
![Lung 3D](<img width="964" height="654" alt="Screenshot 2025-09-29 162536" src="https://github.com/user-attachments/assets/a25920b0-f5d7-4765-b0bc-7eee2ae7a2a1" />

<img width="923" height="559" alt="Screenshot 2025-09-29 162551" src="https://github.com/user-attachments/assets/849ffdd8-6aca-4870-9ed8-ed012ef3cbe4" />


### Pancreas
![Pancreas  3D](![pancreas image ](https://github.com/user-attachments/assets/6cff5847-4a8c-4af7-9cd5-c558ef33b145)
)

## Installation

```bash
pip install -r requirements.txt
