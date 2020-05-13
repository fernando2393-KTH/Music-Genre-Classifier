<<<<<<< Updated upstream
DATASETS = "Datasets/fma_small/"
METADATA = "Datasets/fma_metadata/"
MFCC = "Datasets/mfcc.csv"
=======
# Run locally

DATASETS = "Datasets/fma_small/"
METADATA = "Datasets/fma_metadata/"
MFCC = "Datasets/mfcc.csv"
SPECTROGRAM = "Datasets/spectrogram.csv"
SPECTROGRAM_AUGMENT = "Datasets/spectrogram_augment_1.csv"
AUGMENT = True
"""
# # Run on Google Colab
DATASETS = "/content/drive/My Drive/Datasets/fma_small/"
METADATA = "/content/drive/My Drive/Datasets/fma_metadata/"
MFCC = "/content/drive/My Drive/Datasets/mfcc.csv"
SPECTROGRAM = "/content/drive/My Drive/Datasets/spectrogram.csv"
SPECTROGRAM_AUGMENT = "/content/drive/My Drive/Datasets/spectrogram_augment.csv"
"""

# spectrogram_augment_1.csv: time stretch, rate=1.1
# spectrogram_augment_2.csv: pitch shift, n_steps=2
# spectrogram_augment_3.csv: time stretch, rate=0.9
# spectrogram_augment_4.csv: sampling different 3s from dataset
# spectrogram_augment_5.csv: pitch shift, n_steps=-2

"""
TODO:
    Try rectangular filters
    no augmentation on val and test
    add dropout, batchnormalization
"""
>>>>>>> Stashed changes
