# DT2119 Final Project - Music Genre Classification: Different Methods Exploration

Final project of KTH course DT2119 Speech and Speaker Recognition. This project consists of 
employing different approaches in order to perform music genre classification over the [FMA](https://github.com/mdeff/fma) dataset.

The experiments include from the simplest approaches such as SVM and K-NN to deep learning approaches by means of different
configurations of convolutional neural networks, which can be found in [Models](/Models).

To run the experiments:
- Go to [base_models](base_models.py) to run the simple approaches.
- Go to [final_models](base_models.py) to run the deep learning approaches.
- Modify the paths in [constants](constants.py) if needed (as it is in order to run in Colab).
- Modify the network parameters to customize the architecture and performance.
- Run the selected code (it needs TensorFlow in its version 2 to run).

The conclsusions of the different experiments can be found in [MusicGenreRecognition_Group4](MusicGenreRecognition_Group4.pdf).