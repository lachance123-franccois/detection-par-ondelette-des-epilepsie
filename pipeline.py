import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

from scipy import signal as traitement_signal
from scipy.io import loadmat
import pywt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

import os
import json


class PretraiteurEEG:

    def __init__(self,
                 frequence_echantillonnage = 256,
                 frequence_coupure_basse   = 0.5,
                 frequence_coupure_haute   = 70.0,
                 frequence_reseau          = 50.0,
                 ordre_filtre              = 4):

        self.frequence_echantillonnage = frequence_echantillonnage
        self.frequence_coupure_basse   = frequence_coupure_basse
        self.frequence_coupure_haute   = frequence_coupure_haute
        self.frequence_reseau          = frequence_reseau
        self.ordre_filtre              = ordre_filtre
        self._concevoir_filtres()

    def _concevoir_filtres(self):
        frequence_nyquist = self.frequence_echantillonnage / 2
        freq_normalisee_basse = self.frequence_coupure_basse  / frequence_nyquist
        freq_normalisee_haute = self.frequence_coupure_haute  / frequence_nyquist

        (self.coefficients_b_passebande,
         self.coefficients_a_passebande) = traitement_signal.butter(
            self.ordre_filtre,
            [freq_normalisee_basse, freq_normalisee_haute],
            btype='band'
        )

        freq_reseau_normalisee = self.frequence_reseau / frequence_nyquist
        (self.coefficients_b_notch,
         self.coefficients_a_notch) = traitement_signal.iirnotch(
            freq_reseau_normalisee,
            Q = 30
        )

    def filtrer_signal(self, signal_eeg):
        signal_filtre = traitement_signal.filtfilt(
            self.coefficients_b_passebande,
            self.coefficients_a_passebande,
            signal_eeg
        )

        signal_filtre = traitement_signal.filtfilt(
            self.coefficients_b_notch,
            self.coefficients_a_notch,
            signal_filtre
        )

        return signal_filtre

    def normaliser_zscore(self, signal_eeg):
        moyenne    = signal_eeg.mean()
        ecart_type = signal_eeg.std()
        signal_normalise = (signal_eeg - moyenne) / (ecart_type + 1e-8)

        return signal_normalise

    def pretraiter_tous_segments(self, matrice_signaux):
        segments_pretraites = np.array([
            self.normaliser_zscore(
                self.filtrer_signal(segment)
            )
            for segment in matrice_signaux
        ])

        return segments_pretraites

class AnalyseurTempsFrequence:

    BANDES_CEREBRALES = {
        'delta' : (0.5,  4.0),
        'theta' : (4.0,  8.0),
        'alpha' : (8.0,  13.0),
        'beta'  : (13.0, 30.0),
        'gamma' : (30.0, 70.0),
    }

    def __init__(self,
                 frequence_echantillonnage = 256,
                 taille_fenetre_stft       = 128,
                 recouvrement_stft         = 64,
                 type_ondelette            = 'cmor1.5-1.0'):

        self.frequence_echantillonnage = frequence_echantillonnage
        self.taille_fenetre_stft       = taille_fenetre_stft
        self.recouvrement_stft         = recouvrement_stft
        self.type_ondelette            = type_ondelette

    def calculer_stft(self, signal_eeg):
        vecteur_frequences, vecteur_temps_stft, coefficients_complexes = traitement_signal.stft(
            signal_eeg,
            fs       = self.frequence_echantillonnage,
            window   = 'hann',
            nperseg  = self.taille_fenetre_stft,
            noverlap = self.recouvrement_stft
        )

        matrice_puissance = np.abs(coefficients_complexes) ** 2

        return vecteur_frequences, vecteur_temps_stft, matrice_puissance

    def calculer_cwt(self, signal_eeg, nombre_echelles=64):
        vecteur_frequences_cwt = np.logspace(
            np.log10(0.5),
            np.log10(70),
            nombre_echelles
        )

        echelles_ondelettes = pywt.frequency2scale(
            self.type_ondelette,
            vecteur_frequences_cwt / self.frequence_echantillonnage
        )

        coefficients_cwt, _ = pywt.cwt(
            signal_eeg,
            scales           = echelles_ondelettes,
            wavelet          = self.type_ondelette,
            sampling_period  = 1 / self.frequence_echantillonnage
        )

        matrice_puissance_cwt = np.abs(coefficients_cwt) ** 2

        return vecteur_frequences_cwt, matrice_puissance_cwt

    def calculer_energie_par_bande_stft(self, vecteur_frequences, matrice_puissance_stft):
        energies_par_bande = {}

        for nom_bande, (frequence_basse, frequence_haute) in self.BANDES_CEREBRALES.items():
            masque_bande = (vecteur_frequences >= frequence_basse) & \
                           (vecteur_frequences <= frequence_haute)
            energie_bande = matrice_puissance_stft[masque_bande, :].mean()
            energies_par_bande[nom_bande] = energie_bande

        return energies_par_bande

    def calculer_energie_par_bande_cwt(self, vecteur_frequences_cwt, matrice_puissance_cwt):
        energies_par_bande_cwt = {}

        for nom_bande, (frequence_basse, frequence_haute) in self.BANDES_CEREBRALES.items():
            masque_bande_cwt = (vecteur_frequences_cwt >= frequence_basse) & \
                               (vecteur_frequences_cwt <= frequence_haute)
            energie_bande_cwt = matrice_puissance_cwt[masque_bande_cwt, :].mean()
            energies_par_bande_cwt[nom_bande] = energie_bande_cwt

        return energies_par_bande_cwt

class ExtracteurCaracteristiques:

    def __init__(self, frequence_echantillonnage=256):
        self.analyseur_tf = AnalyseurTempsFrequence(
            frequence_echantillonnage=frequence_echantillonnage
        )

    def calculer_entropie_spectrale(self, matrice_puissance_stft):
        densite_spectrale = matrice_puissance_stft.mean(axis=1)
        densite_normalisee = densite_spectrale / (densite_spectrale.sum() + 1e-12)
        entropie = -np.sum(densite_normalisee * np.log2(densite_normalisee + 1e-12))

        return entropie