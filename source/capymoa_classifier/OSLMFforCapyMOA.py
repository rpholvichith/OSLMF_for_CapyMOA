from capymoa.base import Classifier
import warnings
warnings.filterwarnings("ignore")
import sys
import math
import random
import numpy as np
from time import time
sys.path.append("..")
from sklearn.svm import SVC
from evaluation.helpers import *
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from onlinelearning.ensemble import *
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from onlinelearning.online_learning import *
from onlinelearning.ftrl_adp import FTRL_ADP
from semi_supervised.semiSupervised import *
from sklearn.preprocessing import StandardScaler
from em.online_expectation_maximization import OnlineExpectationMaximization
from schema.schema_methods import from_schema_to_ord_cont

from capymoa.stream import Schema
from capymoa.base import Instance, LabeledInstance
from typing import Literal

class OSLMFforCapyMOA(Classifier):

    def __init__(self, schema: Schema, learner: None | Literal["ftrl_adp"] | Classifier, prediction_mode: Literal["x", "z", "xz"],
                 window_size: int = 200, num_ord_updates: int = 1, decay_coef: float = 0.5, decay_choice: int = 4,
                 contribute_error_rate: float = 0.01, density_percent: int = 5, random_seed: int = 1):
        super().__init__(schema, random_seed)
        # Parameters
        self.prediction_mode = prediction_mode
        self.all_ord_indices, self.all_cont_indices = from_schema_to_ord_cont(schema)
        self.window_size = window_size
        self.n_features = len(self.all_ord_indices)
        self.num_ord_updates = num_ord_updates
        self.decay_coef = decay_coef
        self.density_percent = density_percent
        self.decay_choice = decay_choice
        self.contribute_error_rate = contribute_error_rate

        # Models
        self.oem = OnlineExpectationMaximization(self.all_cont_indices, self.all_ord_indices, window_size=window_size)
        self.clf_x = LinearSVC(random_state=0, tol=1e-5)
        self.clf_z = LinearSVC(random_state=0, tol=1e-5)
        if learner is None:
            self.learner = None
            self.learner_name = None
        elif learner == "ftrl_adp":
            self.learner_name = learner
            self.learner = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=self.n_features)
        else:
            self.learner_name = str(learner)
            self.learner = learner

        # Windowing
        np.random.seed(self.random_seed)
        # Window for original data is initialized to 0
        self.window_x = np.zeros((self.window_size, self.n_features))
        # Window for latent data is initialized to 0 (ordinal) or normal distribution (continuous)
        self.window_z = np.zeros((self.window_size, self.n_features))
        for j in range(self.n_features):
            if self.all_ord_indices[j]:
                self.window_z[:, j].fill(0)
            else:
                self.window_z[:, j] = np.random.normal(0, 1, size=self.window_size)
        self.window_label_x = np.array([np.nan for i in range(self.window_size)])
        self.window_label_z = np.array([np.nan for i in range(self.window_size)])
        self.window_counter = 0

        # Memory
        self.Y_label = []
        self.Y_label_fill_x_ensemble = []
        self.Y_label_fill_z_ensemble = []
        self.Y_label_fill_x = []
        self.Y_label_fill_z = []

    def __str__(self):
        return super().__str__()

    def predict(self, instance: Instance):
        x_masked = instance.x.reshape(1, -1)
        # Imputation of unobserved data
        z_imp, x_imp = self.oem.partial_fit_and_predict(x_masked, max_workers=1, decay_coef=self.decay_coef, sigma_update=False, marginal_update=False)
        # Add imputed data for prediction
        self.window_z[self.window_counter, :] = z_imp
        self.window_x[self.window_counter, :] = x_imp
        self.window_label_z[self.window_counter] = np.nan
        self.window_label_x[self.window_counter] = np.nan
        # Generation of directed graphs
        nneigh_x = DensityPeaks(self.window_x, self.density_percent)
        nneigh_z = DensityPeaks(self.window_z, self.density_percent)
        # Label propagation
        initial_label_x = np.where(~np.isnan(self.window_label_x))[0]
        predict_label_train_x_ensemble = SSC_DensityPeaks_SVC_ensemble(self.window_x, self.window_label_x, self.window_z, self.window_label_z, initial_label_x, nneigh_x, nneigh_z, self.clf_x, self.clf_z)
        initial_label_z = np.where(~np.isnan(self.window_label_z))[0]
        predict_label_train_z_ensemble = SSC_DensityPeaks_SVC_ensemble(self.window_z, self.window_label_z, self.window_x, self.window_label_x, initial_label_z, nneigh_z, nneigh_x, self.clf_z, self.clf_x)
        predict_label_train_x = SSC_DensityPeaks_SVC(self.window_x, self.window_label_x, initial_label_x, nneigh_x, self.clf_x)
        predict_label_train_z = SSC_DensityPeaks_SVC(self.window_z, self.window_label_z, initial_label_z, nneigh_z, self.clf_z)
        # Save propagated labels
        self.Y_label_fill_x_ensemble.append(predict_label_train_x_ensemble[self.window_counter])
        self.Y_label_fill_z_ensemble.append(predict_label_train_z_ensemble[self.window_counter])
        self.Y_label_fill_x.append(predict_label_train_x[self.window_counter])
        self.Y_label_fill_z.append(predict_label_train_z[self.window_counter])
        # Predict with learner (only with x_imp right now)
        if self.learner_name == "ftrl_adp":
            p, decay, loss, w = self.learner.predict([i for i in range(self.n_features)], x_imp)
            y_pred = int(p > 0.5)
        elif self.learner is None:
            y_pred = predict_label_train_x
        else:
            y_pred = self.learner.predict(Instance.from_array(self.schema, x_imp))
        return y_pred
    
    def train(self, instance: LabeledInstance):
        x_masked = instance.x.reshape(1, -1)
        y = instance.y_index
        self.Y_label.append(y)
        # Imputation of unobserved data
        z_imp, x_imp = self.oem.partial_fit_and_predict(x_masked, max_workers=1, decay_coef=self.decay_coef, sigma_update=True, marginal_update=True)
        # Save imputed data
        self.window_z[self.window_counter, :] = z_imp
        self.window_x[self.window_counter, :] = x_imp
        self.window_counter = (self.window_counter + 1) % self.window_size
        # Generation of directed graphs
        nneigh_x = DensityPeaks(self.window_x, self.density_percent)
        nneigh_z = DensityPeaks(self.window_z, self.density_percent)
        # Label propagation
        initial_label_x = np.where(~np.isnan(self.window_label_x))[0]
        predict_label_train_x_ensemble = SSC_DensityPeaks_SVC_ensemble(self.window_x, self.window_label_x, self.window_z, self.window_label_z, initial_label_x, nneigh_x, nneigh_z, self.clf_x, self.clf_z)
        initial_label_z = np.where(~np.isnan(self.window_label_z))[0]
        predict_label_train_z_ensemble = SSC_DensityPeaks_SVC_ensemble(self.window_z, self.window_label_z, self.window_x, self.window_label_x, initial_label_z, nneigh_z, nneigh_x, self.clf_z, self.clf_x)
        predict_label_train_x = SSC_DensityPeaks_SVC(self.window_x, self.window_label_x, initial_label_x, nneigh_x, self.clf_x)
        predict_label_train_z = SSC_DensityPeaks_SVC(self.window_z, self.window_label_z, initial_label_z, nneigh_z, self.clf_z)
        # Train with learner
        if self.learner_name == "ftrl_adp":
            p, decay, loss, w = self.learner.fit([i for i in range(self.n_features)], x_imp, y, self.decay_choice, self.contribute_error_rate)
            y_pred = int(p > 0.5)
        elif self.learner is None:
            y_pred = predict_label_train_x
        else:
            y_pred = self.learner.fit(LabeledInstance.from_array(self.schema, x_imp, y))
        return y_pred
    
    def predict_proba(self, instance: Instance):
        return super().predict_proba(instance)

# if __name__ == '__main__':
#     # dataset: wpbc; ionosphere; wdbc; australian; credit; wbc; diabetes; dna; german; splice; kr_vs_kp; magic04; a8a; stream
#     dataset = "australian"

#     print("Reading data file...", end=" ")
#     # getting hyperparameter
#     contribute_error_rate, window_size_denominator, batch_size_denominator, decay_coef_change, decay_choice, shuffle =\
#         get_cap_hyperparameter(dataset)
#     MASK_NUM = 1
#     if dataset == "Stream1":
#         X = pd.read_csv("../dataset/MaskData/" + dataset + "/X.txt", sep = " ", header = None)
#     else:
#         X = pd.read_csv("../dataset/MaskData/" + dataset + "/X_process.txt", sep = " " , header = None)
#     if dataset == "a8a":
#         X = abs(X)
#     Y_label = pd.read_csv("../dataset/DataLabel/" + dataset + "/Y_label.txt", sep = ' ', header = None)
#     print("Done.")

#     print("Data preprocessing...", end=" ")
#     # ionosphere, diabetes, credit
#     if (dataset == "magic04") & (shuffle == True):
#         shufflenum = shuffle_dataset_1(Y_label)

#     Y_label_masked = random.sample(range(1, Y_label.shape[0]), int(Y_label.shape[0] * 0.5)) # Hard coded 50% of masked label
#     Y_label_masked.sort()
#     Y_label_masked = np.array(Y_label_masked)
#     X_masked = mask_types(X, MASK_NUM, seed = 1)    # For each row, group features by two and mask one randomly
#     X = X.values
#     Y_label = Y_label.values

#     if (dataset == "magic04") & (shuffle == True):
#         X = X[shufflenum]
#         Y_label = Y_label[shufflenum]

#     all_cont_indices = get_cont_indices(X_masked)   # array[bool]: Get indices of features that are considered continuous (at least 14[hardcoded] unique values)
#     all_ord_indices  = ~all_cont_indices            # array[bool]: Get indices of features that are considered discrete

#     n = X_masked.shape[0]
#     feat = X_masked.shape[1]
#     Y_label = Y_label.flatten()
#     Y_label_masked = Y_label_masked.flatten()       # array[int]: Contains mask indices
#     print("Done.")

#     # setting hyperparameter
#     max_iter = batch_size_denominator * 2           # australian: 10*2
#     BATCH_SIZE = math.ceil(n / batch_size_denominator)
#     WINDOW_SIZE = math.ceil(n / window_size_denominator)    # australian: 2
#     NUM_ORD_UPDATES = 1
#     batch_c = 8

#     # start online copula imputation
#     oem = OnlineExpectationMaximization(all_cont_indices, all_ord_indices, window_size=WINDOW_SIZE)
#     j = 0
#     X_imp    = np.empty(X_masked.shape)
#     Z_imp    = np.empty(X_masked.shape)
#     X_masked = np.array(X_masked)
#     Y_label_fill_x = np.empty(Y_label.shape)
#     Y_label_fill_z = np.empty(Y_label.shape)
#     Y_label_fill_x_ensemble = np.empty(Y_label.shape)
#     Y_label_fill_z_ensemble = np.empty(Y_label.shape)

#     clf1 = LinearSVC(random_state=0, tol=1e-5)
#     clf2 = LinearSVC(random_state=0, tol=1e-5)
#     clf_x = LinearSVC(random_state=0, tol=1e-5)
#     clf_z = LinearSVC(random_state=0, tol=1e-5)

#     print("Streaming data for copula learning and label propagation...")
#     while j <= max_iter:    # Hardcoded to two run on the batch
#         start = (j * BATCH_SIZE) % n
#         end   = ((j + 1) * BATCH_SIZE) % n
#         if end < start:
#             indices = np.concatenate((np.arange(end), np.arange(start, n, 1))) # 生成某个范围或区间对应的数组
#         else:
#             indices = np.arange(start, end, 1)
#         if decay_coef_change == 1:                      # Hardcoded as 0 in the helpers.py script right now
#             this_decay_coef = batch_c / (j + batch_c)
#         else:
#             this_decay_coef = 0.5

#         print("Copula calibration", j)
#         # do one step on the batch (= incoming stream of data)
#         Z_imp[indices, :], X_imp[indices, :] = oem.partial_fit_and_predict(X_masked[indices, :], max_workers = 1, decay_coef = this_decay_coef) # Does a step of EM (update sigma, predict new z [ordinal and missing]). Predict missing original data.

#         if start == 0:
#             train_x, label_train_x, initial_label_x = X_imp[indices, :], Y_label[indices, ], Y_label_masked[(Y_label_masked > start) & (Y_label_masked < end) ]
#             train_z, label_train_z, initial_label_z = Z_imp[indices, :], Y_label[indices, ], Y_label_masked[(Y_label_masked > start) & (Y_label_masked < end) ]
#         elif end == 0:
#             train_x, label_train_x, initial_label_x = X_imp[indices, :], Y_label[indices, ], Y_label_masked[(Y_label_masked > start) & (Y_label_masked < ((j + 1) * BATCH_SIZE))] % start   #??? adaptative translation of indices to 0
#             train_z, label_train_z, initial_label_z = Z_imp[indices, :], Y_label[indices, ], Y_label_masked[(Y_label_masked > start) & (Y_label_masked < ((j + 1) * BATCH_SIZE))] % start   #??? adaptative translation of indices to 0
#         elif end < start:
#             initial_label_x = Y_label_masked[(Y_label_masked > start)]
#             initial_label_x = initial_label_x.tolist()
#             initial_label_x.extend(Y_label_masked[(Y_label_masked < end)])
#             initial_label_x = np.array(initial_label_x)
#             np.sort(initial_label_x)
#             initial_label_x = initial_label_x % len(indices)    #??? adaptative translation of indices to start at 0. BUT incorrect implementation
#             train_x, label_train_x = X_imp[indices, :], Y_label[indices,]

#             initial_label_z = Y_label_masked[(Y_label_masked > start)]
#             initial_label_z = initial_label_z.tolist()
#             initial_label_z.extend(Y_label_masked[(Y_label_masked < end)])
#             initial_label_z = np.array(initial_label_z)
#             np.sort(initial_label_z)
#             initial_label_z = initial_label_z % len(indices)    #??? adaptative translation of indices to start at 0. BUT incorrect implementation
#             train_z, label_train_z = Z_imp[indices, :], Y_label[indices,]
#         else:
#             train_x, label_train_x, initial_label_x = X_imp[indices, : ], Y_label[indices, ], Y_label_masked[(Y_label_masked > start) & (Y_label_masked < end) ] % start    #??? adaptative translation of indices to start at 0
#             train_z, label_train_z, initial_label_z = Z_imp[indices, : ], Y_label[indices, ], Y_label_masked[(Y_label_masked > start) & (Y_label_masked < end) ] % start    #??? adaptative translation of indices to start at 0

#         print("Label propagation", j)
#         # preprocess geometric structure (Density Peaks)
#         percent = 5

#         nneigh_x = DensityPeaks(train_x, percent)   # n x 1 matrix: i-th row contains value j. j is the index of the "neighbor" of i. j -> i?
#         nneigh_z = DensityPeaks(train_z, percent)

#         # fit and train classifiers in an ensemble way
#         predict_label_train_x_ensemble = SSC_DensityPeaks_SVC_ensemble(train_x, label_train_x, train_z, label_train_z, initial_label_x, nneigh_x, nneigh_z, clf1, clf2)
#         predict_label_train_z_ensemble = SSC_DensityPeaks_SVC_ensemble(train_z, label_train_z, train_x, label_train_x, initial_label_z, nneigh_z, nneigh_x, clf1, clf2)
#         Y_label_fill_x_ensemble[indices, ] = predict_label_train_x_ensemble
#         Y_label_fill_z_ensemble[indices, ] = predict_label_train_z_ensemble

#         # fit and train classifiers individually
#         predict_label_train_x = SSC_DensityPeaks_SVC(train_x, label_train_x, initial_label_x, nneigh_x, clf_x)
#         Y_label_fill_x[indices,] = predict_label_train_x

#         predict_label_train_z = SSC_DensityPeaks_SVC(train_z, label_train_z, initial_label_z, nneigh_z, clf_z)
#         Y_label_fill_z[indices, ] = predict_label_train_z

#         j += 1
#     print("Done.")

#     # geting and drawing the error of Semi-Supervised
#     Cumulative_error_rate_semi(Y_label_fill_x_ensemble,
#                                Y_label_fill_z_ensemble,
#                                Y_label_fill_x,
#                                Y_label_fill_z,
#                                Y_label,
#                                dataset)

#     temp = np.ones((n, 1))
    
#     X_masked = pd.DataFrame(X_masked)
#     X_zeros = X_masked.fillna(value=0)
#     X_input_zero = np.hstack((temp, X_zeros))
#     X_input_imp = np.hstack((temp, X_imp))      # added to replace X_zeros
#     if shuffle == True:
#         perm = np.arange(n)
#         np.random.seed(1)
#         np.random.shuffle(perm)
#         Y_label = Y_label[perm]
#         X_input_zero = X_input_zero[perm]
#         X_input_imp = X_input_imp[perm]
#         Y_label_fill_x = Y_label_fill_x[perm]
#         Y_label_fill_z = Y_label_fill_z[perm]

#     print("Streaming data for label prediction...")
#     batch = math.ceil(n / 100)

#     print("\twith original data only...")
#     #X_Zero_CER_fill, svm_error = generate_X_Y(n, X_input_zero, Y_label_fill_x, Y_label, decay_choice, contribute_error_rate)
#     X_Zero_CER_fill, svm_error = generate_X_Y(n, X_input_imp, Y_label_fill_x, Y_label, decay_choice, contribute_error_rate)

#     #X_Zero_CER = generate_Xmask(n, X_input_zero, Y_label, Y_label_masked, decay_choice, contribute_error_rate)
#     X_Zero_CER = generate_Xmask(n, X_input_imp, Y_label, Y_label_masked, decay_choice, contribute_error_rate)

#     #get the error of latent space
#     temp_zim = np.ones((n, 1))
#     X_input_z_imp = np.hstack((temp, Z_imp))
#     if shuffle == True:
#         perm = np.arange(n)
#         np.random.seed(1)
#         np.random.shuffle(perm)
#         X_input_z_imp = X_input_z_imp[perm]

#     print("\twith latent representations...")
#     Z_impl_CER = generate_Xmask(n, X_input_z_imp, Y_label, Y_label_masked, decay_choice, contribute_error_rate)
#     Z_impl_CER_fill, svm_error_z = generate_X_Y(n, X_input_z_imp, Y_label_fill_z, Y_label, decay_choice, contribute_error_rate)

#     # ensemble_XZ_imp_CER_fill, lamda_array_XZ_fill = ensemble_Y(n, X_input_z_imp, X_input_zero, Y_label, Y_label_fill_x, decay_choice, contribute_error_rate)

#     # ensemble_XZ_imp_CER = ensemble_Xmask(n, X_input_z_imp, X_input_zero, Y_label, Y_label_masked, decay_choice, contribute_error_rate)

#     ensemble_XZ_imp_CER_fill, lamda_array_XZ_fill = ensemble_Y(n, X_input_z_imp, X_input_imp, Y_label, Y_label_fill_x, decay_choice, contribute_error_rate)

#     ensemble_XZ_imp_CER = ensemble_Xmask(n, X_input_z_imp, X_input_imp, Y_label, Y_label_masked, decay_choice, contribute_error_rate)
#     print("Done.")

#     draw_cap_error_picture(ensemble_XZ_imp_CER_fill,
#                            X_Zero_CER_fill,
#                            ensemble_XZ_imp_CER,
#                            X_Zero_CER,
#                            Z_impl_CER,
#                            Z_impl_CER_fill,
#                            svm_error,
#                            dataset)
