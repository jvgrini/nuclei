import numpy as np
import os
from skimage import io, filters, measure, morphology, exposure, segmentation, feature
from scipy.ndimage import distance_transform_edt, label
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, jaccard_score
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

def custom_evaluation_metric(y_true, y_pred):
    y_true_flat = np.concatenate([arr.flatten() for arr in y_true])
    y_pred_flat = np.concatenate([arr.flatten() for arr in y_pred])
    if len(np.unique(y_true_flat)) > 2:
        jaccard_index = jaccard_score(y_true_flat, y_pred_flat, average='weighted')
    else:
        jaccard_index = jaccard_score(y_true_flat, y_pred_flat)
    return jaccard_index

class SegmentationModel(BaseEstimator, TransformerMixin):
    def __init__(self, sigma=1.0, clip_limit=0.02, kernel_size=3, opening_radius=2, closing_radius=2, min_region_size=500, footprint=5, footprint_z=1, kernel_thresh=11):
        self.sigma = sigma
        self.clip_limit = clip_limit
        self.kernel_size = kernel_size
        self.opening_radius = opening_radius
        self.closing_radius = closing_radius
        self.min_region_size = min_region_size
        self.footprint = footprint
        self.footprint_z = footprint_z
        self.kernel_thresh = kernel_thresh

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        predictions = []
        for image in X:
            predictions.append(self.transform(image))
        return predictions

    def transform(self, X):
        filtered = filters.gaussian(X, sigma=self.sigma)
        equalized_channel = exposure.equalize_adapthist(filtered, kernel_size=self.kernel_size, clip_limit=self.clip_limit)

        binary_volume = (equalized_channel > filters.threshold_local(equalized_channel, block_size=self.kernel_thresh, method='gaussian', offset=0))
        binary_volume = morphology.binary_opening(binary_volume, morphology.ball(radius=self.opening_radius))            
        binary_volume = morphology.binary_closing(binary_volume, morphology.ball(radius=self.closing_radius))

        labeled_volume = measure.label(binary_volume)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filtered_labeled_volume = morphology.remove_small_objects(labeled_volume, min_size=self.min_region_size)
        distance = distance_transform_edt(filtered_labeled_volume, sampling=(1,1,2.6823))            
        coords = feature.peak_local_max(distance, footprint=np.ones([self.footprint, self.footprint, self.footprint_z]), labels=filtered_labeled_volume)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = label(mask)
        labels = segmentation.watershed(-distance, markers, mask=filtered_labeled_volume)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = morphology.remove_small_objects(labels, min_size=self.min_region_size)

        return labels

param_grid = {
    'sigma': [2.0],
    'clip_limit': [0.01],
    'kernel_size': [5, 7],
    'opening_radius': [1, 2, 3],
    'closing_radius': [1, 2, 3],
    'min_region_size': [400],
    'footprint_z': [3,5,7],
    'footprint': [5,7,9,11,13],
    'kernel_thresh': [251],
}

kf = KFold(n_splits=2, shuffle=True, random_state=42)

X_folder = 'X'
y_folder = 'y'

X_file_names = sorted(os.listdir(X_folder))
y_file_names = sorted(os.listdir(y_folder))


segmentation_model = SegmentationModel()

# Create a GridSearchCV object with the segmentation model and parameter grid
grid_search = GridSearchCV(
    segmentation_model,
    param_grid,
    scoring=make_scorer(custom_evaluation_metric),
    cv=kf,
    n_jobs=-1
)


# Iterate through the folds
for fold, (train_index, test_index) in enumerate(kf.split(X_file_names)):
    train_files = [X_file_names[i] for i in train_index]
    test_files = [X_file_names[i] for i in test_index]

    # Load and print shapes of images for training
    X_train = [io.imread(os.path.join(X_folder, file_name)) for file_name in train_files]

    # Load and print shapes of corresponding labels for training
    y_train_files = ['y' + file_name[1:] for file_name in train_files]
    y_train = [io.imread(os.path.join(y_folder, file_name)) for file_name in y_train_files]

    # Load and print shapes of images and labels for testing (similar modifications)
    X_test = [io.imread(os.path.join(X_folder, file_name)) for file_name in test_files]
    y_test_files = ['y' + file_name[1:] for file_name in test_files]
    y_test = [io.imread(os.path.join(y_folder, file_name)) for file_name in y_test_files]

    # Fit the grid search to your data
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best Parameters (Fold {fold + 1}): {best_params}")

    # Evaluate the model with the best parameters on the test set
    best_model = grid_search.best_estimator_
    
    # Ensure the model has a 'predict' method
    if hasattr(best_model, 'predict'):
        # Make predictions on the test set
        y_pred = best_model.predict(X_test)
        
        # Evaluate the model with the custom evaluation metric
        test_score = custom_evaluation_metric(y_test, y_pred)
        print(f"Test Score (Jaccard index) (Fold {fold + 1}): {test_score}")
    else:
        print("Model does not have a 'predict' method.")
