
import numpy as np
from scipy import signal as traitement_signal
import pywt

from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


class GenerateurDonneesEEG:
    def __init__(self, frequence_echantillonnage=256, duree_segment=4.0, graine_aleatoire=42):
        self.fs = frequence_echantillonnage
        self.n = int(frequence_echantillonnage * duree_segment)
        self.t = np.linspace(0, duree_segment, self.n, endpoint=False)
        self.rng = np.random.default_rng(graine_aleatoire)

    def generer_signal_interictal(self):
        s = 0.0
        for amplitude, frequence in [(0.8, 2), (0.5, 6), (1.2, 10), (0.3, 20)]:
            s = s + amplitude * np.sin(2 * np.pi * frequence * self.t
                                       + self.rng.uniform(0, 2 * np.pi))
        return s + 0.4 * self.rng.standard_normal(self.n)

    def generer_signal_ictal(self):
        fond = self.generer_signal_interictal() * 0.3
        gamma_init = 2.5 * np.sin(2 * np.pi * 40 * self.t) * np.exp(-self.t * 0.3)
        gamma_sout = 1.8 * np.sin(2 * np.pi * 55 * self.t + 0.5) * (1 - np.exp(-self.t * 0.8))
        pointe_onde = 3.0 * np.sin(2 * np.pi * 3 * self.t) * np.abs(np.sin(2 * np.pi * 3 * self.t))
        enveloppe = np.linspace(0.5, 2.0, self.n)
        return (fond + gamma_init + gamma_sout + pointe_onde) * enveloppe

    def construire_jeu_de_donnees(self, nb_normaux=150, nb_crise=60):
        X = [self.generer_signal_interictal() for _ in range(nb_normaux)]
        X += [self.generer_signal_ictal() for _ in range(nb_crise)]
        y = [0] * nb_normaux + [1] * nb_crise
        return np.array(X), np.array(y)


class PretraiteurEEG:
    def __init__(self, fs=256, f_basse=0.5, f_haute=70.0, f_reseau=50.0, ordre=4):
        self.fs = fs
        nyq = fs / 2
        self.sos_bp = traitement_signal.butter(
            ordre, [f_basse / nyq, f_haute / nyq], btype="band", output="sos")
        b, a = traitement_signal.iirnotch(f_reseau / nyq, Q=30)
        self.sos_notch = traitement_signal.tf2sos(b, a)

    def filtrer_signal(self, x):
        x = traitement_signal.sosfiltfilt(self.sos_bp, x)
        return traitement_signal.sosfiltfilt(self.sos_notch, x)

    def normaliser_zscore(self, x):
        return (x - x.mean()) / (x.std() + 1e-12)

    def pretraiter_tous_segments(self, X):
        return np.array([self.normaliser_zscore(self.filtrer_signal(seg)) for seg in X])


class AnalyseurTempsFrequence:
    BANDES = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
              "beta": (13.0, 30.0), "gamma": (30.0, 70.0)}

    def __init__(self, fs=256, nperseg=128, noverlap=64, ondelette="cmor1.5-1.0"):
        self.fs, self.nperseg, self.noverlap, self.ondelette = fs, nperseg, noverlap, ondelette

    def calculer_stft(self, x):
        f, t, Z = traitement_signal.stft(x, fs=self.fs, window="hann",
                                         nperseg=self.nperseg, noverlap=self.noverlap)
        return f, t, np.abs(Z) ** 2

    def calculer_cwt(self, x, n_echelles=64):
        """ATTENTION : c'est une CWT (ondelette de Morlet complexe), PAS une DWT."""
        f = np.logspace(np.log10(0.5), np.log10(70), n_echelles)
        echelles = pywt.frequency2scale(self.ondelette, f / self.fs)
        coefs, _ = pywt.cwt(x, scales=echelles, wavelet=self.ondelette,
                            sampling_period=1 / self.fs)
        return f, np.abs(coefs) ** 2

    def energie_par_bande(self, f, P):
        return {nom: P[(f >= lo) & (f <= hi), :].mean()
                for nom, (lo, hi) in self.BANDES.items()}


class ExtracteurCaracteristiques:


    NOMS = (["stft_" + b for b in AnalyseurTempsFrequence.BANDES]
            + ["cwt_" + b for b in AnalyseurTempsFrequence.BANDES]
            + ["entropie_spectrale", "aplatissement", "variation_amplitude"])

    def __init__(self, fs=256):
        self.tf = AnalyseurTempsFrequence(fs)

    def calculer_entropie_spectrale(self, P):
       
        dsp = P.mean(axis=1)
        p = dsp / (dsp.sum() + 1e-12)
        return -np.sum(p * np.log2(p + 1e-12))

    def extraire_un_segment(self, x):
        f_stft, _, P_stft = self.tf.calculer_stft(x)
        f_cwt, P_cwt = self.tf.calculer_cwt(x)

        e_stft = self.tf.energie_par_bande(f_stft, P_stft)
        e_cwt = self.tf.energie_par_bande(f_cwt, P_cwt)

        moitie = len(x) // 2
        variation = x[moitie:].std() / (x[:moitie].std() + 1e-8)  # capte l'enveloppe croissante

        # Invariant a l'echelle (contrairement a x.std()) => survit au z-score
        aplatissement = ((x - x.mean()) ** 4).mean() / (x.var() ** 2 + 1e-12)

        return np.array(
            [e_stft[b] for b in AnalyseurTempsFrequence.BANDES]
            + [e_cwt[b] for b in AnalyseurTempsFrequence.BANDES]
            + [self.calculer_entropie_spectrale(P_stft), aplatissement, variation]
        )

    def extraire_tous(self, X):
        return np.array([self.extraire_un_segment(seg) for seg in X])

    @classmethod
    def index_de(cls, nom):

        return cls.NOMS.index(nom)

    @staticmethod
    def controler_descripteurs(F, seuil_variance=1e-10):

        variances = F.var(axis=0)
        morts = [ExtracteurCaracteristiques.NOMS[i]
                 for i, v in enumerate(variances) if v < seuil_variance]
        if morts:
            print(f"  [ALERTE] descripteur(s) sans information : {morts}")
        else:
            print("  [OK] tous les descripteurs ont une variance non nulle")
        return morts


def pipeline_svm(seed=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        # CORRECTION 4 : jeu desequilibre (150 vs 60)
        ("svm", SVC(kernel="rbf", probability=True,
                    class_weight="balanced", random_state=seed)),
    ])


def validation_imbriquee(X, y, seed=42):

    grille = {"svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", "auto", 0.01, 0.1]}
    interne = GridSearchCV(pipeline_svm(seed), grille,
                           cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                           scoring="f1", n_jobs=-1)
    externe = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_score(interne, X, y, cv=externe, scoring="f1")


if __name__ == "__main__":
    X_brut, y = GenerateurDonneesEEG().construire_jeu_de_donnees()
    X_propre = PretraiteurEEG().pretraiter_tous_segments(X_brut)
    F = ExtracteurCaracteristiques().extraire_tous(X_propre)

    print(f"Matrice de features : {F.shape}   (segments x descripteurs)")
    print(f"Classes : {np.bincount(y)}  -> {100 * y.mean():.0f} % de crises")
    ExtracteurCaracteristiques.controler_descripteurs(F)
    print()

    scores = validation_imbriquee(F, y)
    print(f"F1 (CV imbriquee)      : {scores.mean():.4f} +/- {scores.std():.4f}")

    naif = cross_val_score(DummyClassifier(strategy="most_frequent"), F, y,
                           cv=StratifiedKFold(5, shuffle=True, random_state=42),
                           scoring="f1")
    print(f"F1 (modele naif)       : {naif.mean():.4f}")
    for nom in ("variation_amplitude", "aplatissement", "cwt_gamma"):
        j = ExtracteurCaracteristiques.index_de(nom)
        s = cross_val_score(pipeline_svm(), F[:, [j]], y,
                            cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            scoring="f1")
        print(f"F1 ({nom:<20s} SEUL) : {s.mean():.4f}")

    modele = pipeline_svm().fit(F, y)
    print("\n" + classification_report(y, modele.predict(F),
                                       target_names=["Normal", "Crise"]))