# AudioClassifier
Final project for Berkeley's FSDL 2021 course  
Audio classification training library with a streamlit app for evaluations & inferencing.  

The project builds upon knowledge, practices & code gathered from the following list of resources:
- Full Stack Deep Learning materials - https://fullstackdeeplearning.com/
- Made With ML "MLOps" series - https://madewithml.com/courses/mlops/
- Mikes Males's Sound Classification project - https://mikesmales.medium.com/sound-classification-using-deep-learning-8bc2aa1990b7
- Fast.ai's course & library - https://docs.fast.ai/

Implementation was done using Jupytext, to allow the comfort of working with notebooks.  

Repository content introduction:
- create_dataset.py - will preprocess the audio samples into a tabular dataset based on [MFCC features](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
- pipeline.py - notebook (after using jupytext of course) that illustrates the process of EDAing our data, performing and evaluating the different experiments
- inference.py - our streamlit app. Exposes EDA, evaluations & inferencing utilities.
- libraries
    - data.py - data-related operations
    - model.py - model and analysis-related operations