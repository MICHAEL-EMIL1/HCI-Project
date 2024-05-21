import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import pywt
import statsmodels.api as sm
import scipy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# preprocessing
# Function to apply bandpass filter to ECG signal
def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Function to perform baseline correction on ECG signal
def baseline_correction(signal):
    baseline = np.mean(signal[:100])  # Use the mean of the first 100 samples as baseline
    corrected_signal = signal - baseline
    return corrected_signal

# Function to detect R-peaks in ECG signal using find_peaks from scipy
def detect_r_peaks(signal, fs):
    peaks, _ = find_peaks(signal, distance=int(fs * 0.4), prominence=0.3)  # Adjust distance based on expected R-R interval
    return peaks

# Function to segment heartbeats based on R-peaks
def segment_heartbeats(signal, rpeaks_indices, window_size):
    heartbeats = []
    for idx in rpeaks_indices:
        start = idx - window_size // 2
        end = idx + window_size // 2
        heartbeat = signal[start:end]
        heartbeats.append(heartbeat)
    return heartbeats

# Function to calculate AC\DCT features from a segmented heartbeat
def extract_asdct_features(signal):
    acf = sm.tsa.acf(signal, nlags=len(signal) - 1)
    dct = scipy.fftpack.dct(acf, type=2)
    return dct

# Function to calculate Wavelet coefficients from a segmented heartbeat
def extract_wavelet_features(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    wavelet_features = np.hstack(coeffs)
    return wavelet_features

# Load and process ECG data with specified feature extraction method for AC\DCT
def load_process_data_asdct(directory, num_rows_to_read=30000, max_features_length=100):
    file_list = [file for file in os.listdir(directory) if file.endswith('.csv')]
    data = []
    labels = []

    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        df = pd.read_csv(file_path, nrows=num_rows_to_read)
        ecg_signal = df[df.columns[9]].values  # Assuming ECG signal is in the first column

        # Preprocessing steps
        fs = 1000.0  # Sampling frequency
        filtered_signal = bandpass_filter(ecg_signal, fs)
        baseline_corrected_signal = baseline_correction(filtered_signal)
        rpeaks_indices = detect_r_peaks(baseline_corrected_signal, fs)
        heartbeats = segment_heartbeats(baseline_corrected_signal, rpeaks_indices, window_size=400)

        for heartbeat in heartbeats:
            if len(heartbeat) > 0:
                features = extract_asdct_features(heartbeat)

                # Pad or truncate feature vector to fixed length
                if len(features) < max_features_length:
                    padded_features = np.pad(features, (0, max_features_length - len(features)), mode='constant')
                else:
                    padded_features = features[:max_features_length]

                data.append(padded_features)
                labels.append(int(file_name.split('.')[0].replace('Patient', '')))

    return np.array(data), np.array(labels)

# Load and process ECG data with specified feature extraction method for Wavelet
def load_process_data_wavelet(directory, num_rows_to_read=30000, max_features_length=100):
    file_list = [file for file in os.listdir(directory) if file.endswith('.csv')]
    data = []
    labels = []

    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        df = pd.read_csv(file_path, nrows=num_rows_to_read)
        ecg_signal = df[df.columns[9]].values  # Assuming ECG signal is in the first column

        # Preprocessing steps
        fs = 1000.0  # Sampling frequency
        filtered_signal = bandpass_filter(ecg_signal, fs)
        baseline_corrected_signal = baseline_correction(filtered_signal)
        rpeaks_indices = detect_r_peaks(baseline_corrected_signal, fs)
        heartbeats = segment_heartbeats(baseline_corrected_signal, rpeaks_indices, window_size=400)

        for heartbeat in heartbeats:
            if len(heartbeat) > 0:
                features = extract_wavelet_features(heartbeat)

                # Pad or truncate feature vector to fixed length
                if len(features) < max_features_length:
                    padded_features = np.pad(features, (0, max_features_length - len(features)), mode='constant')
                else:
                    padded_features = features[:max_features_length]

                data.append(padded_features)
                labels.append(int(file_name.split('.')[0].replace('Patient', '')))

    return np.array(data), np.array(labels)

# Train and evaluate classifier with specified feature type and perform GridSearchCV for hyperparameter tuning
def train_evaluate_classifier(data, labels, classifier_type='svm', param_grid=None, plot_confusion_matrix=False):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.35, random_state=42)

    if classifier_type == 'svm':
        classifier = SVC()
    elif classifier_type == 'random_forest':
        classifier = RandomForestClassifier()

    if param_grid:
        grid_search = GridSearchCV(classifier, param_grid, cv=5)  # 5-fold cross-validation
        grid_search.fit(X_train, y_train)
        best_classifier = grid_search.best_estimator_
    else:
        best_classifier = classifier

    best_classifier.fit(X_train, y_train)
    y_train_pred = best_classifier.predict(X_train)
    y_test_pred = best_classifier.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Compute confusion matrix and display
    cm = confusion_matrix(y_test, y_test_pred)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    # plt.title(f'Confusion Matrix - {classifier_type} Classifier')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.show()

    print(f"Training Accuracy with {classifier_type} Classifier: {train_accuracy:.4f}")
    print(f"Test Accuracy with {classifier_type} Classifier: {test_accuracy:.4f}")

    return test_accuracy, best_classifier

def identify_subject(classifiers, input_file, directory):
    # Load and preprocess the input file
    df = pd.read_csv(input_file)
    ecg_signal = df[df.columns[9]].values  # Assuming ECG signal is in the tenth column

    # Preprocessing steps
    fs = 1000.0  # Sampling frequency
    filtered_signal = bandpass_filter(ecg_signal, fs)
    baseline_corrected_signal = baseline_correction(filtered_signal)
    rpeaks_indices = detect_r_peaks(baseline_corrected_signal, fs)
    heartbeats = segment_heartbeats(baseline_corrected_signal, rpeaks_indices, window_size=400)

    # Extract features from segmented heartbeats
    features = []
    for heartbeat in heartbeats:
        if len(heartbeat) > 0:
            features.append(extract_asdct_features(heartbeat))  # Use the same feature extraction method as your subjects

    # Pad or truncate feature vectors to fixed length
    max_features_length = 100  # Adjust as needed
    for i, feat in enumerate(features):
        if len(feat) < max_features_length:
            padded_features = np.pad(feat, (0, max_features_length - len(feat)), mode='constant')
            features[i] = padded_features
        else:
            features[i] = feat[:max_features_length]

    features = np.array(features)
    accuracies = []
    predicted_labels_list = []
    # Test the input file with each classifier
    for classifier in classifiers:
        predicted_labels = classifier.predict(features)
        for predicted_label in predicted_labels:
            print(f"Predicted Label: {predicted_label}")

            # If the predicted label matches any of the subjects and accuracy is above 0.5, print the subject's photo
            photo_path = os.path.join(directory, 'Photos', f"Patient{predicted_label}.jpg")
            if os.path.exists(photo_path):
                return os.path.join(directory, 'Photos', f"Patient{predicted_label}.jpg")
            else:
                print(f"Photo not found for subject: Patient{predicted_label}")
                predicted_labels_list.append(predicted_label)
            break
        else:
            continue  # Continue to the next classifier if no match is found


        # Calculate accuracy as the fraction of correctly predicted labels
        accuracy = np.mean(predicted_labels == predicted_label)  # Assuming the last predicted label is correct
        accuracies.append(accuracy)

        break
    else:
        print("Subject not found or accuracy is below 0.5.")
        return 0

    # Print prediction accuracies
    for i, classifier in enumerate(classifiers):
        print(f"{classifier.__class__.__name__} ")
        print(f"i: {i}")
        print(f"classifier{classifier}")
        print(f"Prediction Accuracy: {accuracies[i]*100:.4f}%")
        #Prediction Accuracy: {accuracies[i]*100:.2f}%")

    # Return predicted labels and accuracies
    return predicted_labels_list, accuracies


# Main function to run the project
def main():
    # Directory containing the CSV files
    directory = 'W:\\youss\\UNI\\10\\HCI\\hci_project\\HCI_Data'

    # Load and process data with AS\DCT features
    asdct_data, labels = load_process_data_asdct(directory)

    # Define parameter grid for SVM and Random Forest classifiers
    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.01, 0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly']
    }

    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Train and evaluate SVM classifier with AS\DCT features and perform GridSearchCV
    svm_asdct_accuracy, svm_asdct_classifier = train_evaluate_classifier(asdct_data, labels, classifier_type='svm', param_grid=svm_param_grid, plot_confusion_matrix=False)

    # Train and evaluate Random Forest classifier with AS\DCT features and perform GridSearchCV
    rf_asdct_accuracy, rf_asdct_classifier = train_evaluate_classifier(asdct_data, labels, classifier_type='random_forest', param_grid=rf_param_grid, plot_confusion_matrix=False)

    # Load and process data with Wavelet features
    wavelet_data, labels = load_process_data_wavelet(directory)

    # Train and evaluate SVM classifier with Wavelet features and perform GridSearchCV
    svm_wavelet_accuracy, svm_wavelet_classifier = train_evaluate_classifier(wavelet_data, labels, classifier_type='svm', param_grid=svm_param_grid, plot_confusion_matrix=False)

    # Train and evaluate Random Forest classifier with Wavelet features and perform GridSearchCV
    rf_wavelet_accuracy, rf_wavelet_classifier = train_evaluate_classifier(wavelet_data, labels, classifier_type='random_forest', param_grid=rf_param_grid, plot_confusion_matrix=False)

    print("\nSummary of Classifier Accuracies:")
    print(f"SVM Accuracy with AS\DCT features: {svm_asdct_accuracy:.4f}")
    print(f"Random Forest Accuracy with AS\DCT features: {rf_asdct_accuracy:.4f}")
    print(f"SVM Accuracy with Wavelet features: {svm_wavelet_accuracy:.4f}")
    print(f"Random Forest Accuracy with Wavelet features: {rf_wavelet_accuracy:.4f}")

    # Subject Identification
    f_p_1 = "W:\\youss\\UNI\\10\\HCI\\hci_project\\HCI_Data\\Patient1.csv"
    f_p_2 = "W:\\youss\\UNI\\10\\HCI\\hci_project\\HCI_Data\\Patient2.csv"
    f_p_3 = "W:\\youss\\UNI\\10\\HCI\\hci_project\\HCI_Data\\Patient3.csv"
    f_p_4 = "W:\\youss\\UNI\\10\\HCI\\hci_project\\HCI_Data\\Patient4.csv"
    f_dummy = "W:\\youss\\UNI\\10\\HCI\\hci_project\\HCI_Data\\s0001_reee.csv"
    identify_subject([svm_asdct_classifier, rf_asdct_classifier, svm_wavelet_classifier, rf_wavelet_classifier], f_p_2, directory)

def identify(csv_file):
    # Directory containing the CSV files
    directory = 'W:\\youss\\UNI\\10\\HCI\\hci_project\\HCI_Data'

    # Load and process data with AS\DCT features
    asdct_data, labels = load_process_data_asdct(directory)

    # Define parameter grid for SVM and Random Forest classifiers
    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.01, 0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly']
    }

    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Train and evaluate SVM classifier with AS\DCT features and perform GridSearchCV
    svm_asdct_accuracy, svm_asdct_classifier = train_evaluate_classifier(asdct_data, labels, classifier_type='svm',
                                                                         param_grid=svm_param_grid,
                                                                         plot_confusion_matrix=False)

    # Train and evaluate Random Forest classifier with AS\DCT features and perform GridSearchCV
    rf_asdct_accuracy, rf_asdct_classifier = train_evaluate_classifier(asdct_data, labels,
                                                                       classifier_type='random_forest',
                                                                       param_grid=rf_param_grid,
                                                                       plot_confusion_matrix=False)

    # Load and process data with Wavelet features
    wavelet_data, labels = load_process_data_wavelet(directory)

    # Train and evaluate SVM classifier with Wavelet features and perform GridSearchCV
    svm_wavelet_accuracy, svm_wavelet_classifier = train_evaluate_classifier(wavelet_data, labels,
                                                                             classifier_type='svm',
                                                                             param_grid=svm_param_grid,
                                                                             plot_confusion_matrix=False)

    # Train and evaluate Random Forest classifier with Wavelet features and perform GridSearchCV
    rf_wavelet_accuracy, rf_wavelet_classifier = train_evaluate_classifier(wavelet_data, labels,
                                                                           classifier_type='random_forest',
                                                                           param_grid=rf_param_grid,
                                                                           plot_confusion_matrix=False)

    print("\nSummary of Classifier Accuracies:")
    print(f"SVM Accuracy with AS\DCT features: {svm_asdct_accuracy:.4f}")
    print(f"Random Forest Accuracy with AS\DCT features: {rf_asdct_accuracy:.4f}")
    print(f"SVM Accuracy with Wavelet features: {svm_wavelet_accuracy:.4f}")
    print(f"Random Forest Accuracy with Wavelet features: {rf_wavelet_accuracy:.4f}")

    identify_subject([svm_asdct_classifier, rf_asdct_classifier, svm_wavelet_classifier, rf_wavelet_classifier], csv_file,
                     directory)


