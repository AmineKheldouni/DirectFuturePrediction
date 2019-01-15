# Object Recognition Project - Direct Future Prediction model
Code for the paper "Learning to Act by Predicting the Future" using Mr Felix Yu's implementation of DFP and adapting it for our approach (https://github.com/flyyufelix/Direct-Future-Prediction-Keras)


## Requierements
- Python Basics (NumPy, matplotlib ...)
- Pandas
- Scikit-image
- tensorflow
- keras
- vizdoom
- FCRN Depth Predictor (https://github.com/iro-cp/FCRN-DepthPrediction)
- Matterport's Mask R-CNN (https://github.com/matterport/Mask_RCNN)


## Running

In DFP file, run the following command in terminal:
```
python dfp_extended_measures.py title nb_measures add_depth_map test_phase environment_type train_random_goal
```
