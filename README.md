# Tutorial of: Deep Learning from Diffuse Biomedical Optics Time-Series: An fNIRS-Focused Review of Recent Advancements and Future Directions.

Creating the environments and downloading the datasets. 
- Install Cedalion from our official [repository](https://github.com/ibs-lab/cedalion)
- Download the datasets:
  - BallSqueezing-HD from here. Please refer to the official publication for more details.
  - Mental Arrythmatic Multi-Model dataset. Please refer to the official publication for more details.
- Install PyTorch with or without cuda-support

Deep Learning model development pipeline.
- To create the dataset, use the preprocessing.py code segment. Modify the directory information accordingly to point to the downloaded dataset locations.
- By default, preprocessing.py will create a dataset without augmentation (0.5Hz), run the code again with different frequencies by modifying FMAX variable. In the paper, we used 0.7 and 1.0. 
- Use the sessions.py file, only for BSQ-HD dataset to obtain the session specific train/validation split.  
- Each folder contain specific script to train deep learning models (cvloso_*.py). Each of them correspond to the experiment types (e, e.f, e.t and e.f.t)
- Modify the model traning codes accordignly with the directory/file information. You will need to make sure the dataset folders, and any additional files are within the correct directory structure.
- To save the additional meta information, create models and loss folder. Please checkout the code, we are saving the performance stats per-each epoch and the model is saved based on the validaiton losses. 
