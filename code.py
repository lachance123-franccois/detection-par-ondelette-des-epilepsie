"""
Detection de crises epileptiques sur EEG synthetique.

Chaine complete : generation -> pretraitement (Butterworth + notch + z-score)
-> analyse temps-frequence (STFT + CWT de Morlet) -> 14 descripteurs -> SVM RBF
-> validation croisee imbriquee.

ATTENTION : les donnees sont SYNTHETIQUES. Les scores obtenus (F1 ~ 1.0)
mesurent la coherence du pipeline, pas une performance clinique.
"""

import warnings

import numpy as np
import pywt
from scipy import signal as traitement_signal

from sklearn.dummy import DummyClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# On ne masque que l'avertissement attendu du classifieur naif
# (F1 non defini quand la classe positive n'est jamais predite).
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

EPS = 1e-12


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
        if not 0 < f_basse < f_haute < nyq:
            raise ValueError(f"Bande [{f_basse}, {f_haute}] Hz incompatible avec fs={fs} Hz.")

        self.sos_bp = traitement_signal.butter(
            ordre, [f_basse / nyq, f_haute / nyq], btype="band", output="sos")
        b, a = traitement_signal.iirnotch(f_reseau / nyq, Q=30)
        self.sos_notch = traitement_signal.tf2sos(b, a)

    def filtrer_signal(self, x):
        x = traitement_signal.sosfiltfilt(self.sos_bp, x)
        return traitement_signal.sosfiltfilt(self.sos_notch, x)

    def normaliser_zscore(self, x):
        return (x - x.mean()) / (x.std() + EPS)

    def pretraiter_tous_segments(self, X):
        return np.array([self.normaliser_zscore(self.filtrer_signal(seg)) for seg in X])


class AnalyseurTempsFrequence:
    BANDES = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
              "beta": (13.0, 30.0), "gamma": (30.0, 70.0)}

    def __init__(self, fs=256, nperseg=256, noverlap=128, ondelette="cmor1.5-1.0"):
        self.fs, self.nperseg, self.noverlap, self.ondelette = fs, nperseg, noverlap, ondelette

    def calculer_stft(self, x):
        f, t, Z = traitement_signal.stft(x, fs=self.fs, window="hann",
                                         nperseg=self.nperseg, noverlap=self.noverlap)
        return f, t, np.abs(Z) ** 2

    def calculer_cwt(self, x, n_echelles=64):
        f = np.logspace(np.log10(0.5), np.log10(70), n_echelles)
        echelles = pywt.frequency2scale(self.ondelette, f / self.fs)
        coefs, _ = pywt.cwt(x, scales=echelles, wavelet=self.ondelette,
                            sampling_period=1 / self.fs)
        return f, np.abs(coefs) ** 2

    def energie_par_bande(self, f, P):
        energies, derniere = {}, list(self.BANDES)[-1]
        for nom, (lo, hi) in self.BANDES.items():
            masque = (f >= lo) & (f <= hi) if nom == derniere else (f >= lo) & (f < hi)
            if not masque.any():
                raise ValueError(
                    f"Bande '{nom}' vide : resolution frequentielle insuffisante "
                    f"(df = {f[1] - f[0]:.2f} Hz). Augmenter nperseg ou n_echelles.")
            energies[nom] = P[masque, :].mean()
        return energies


class ExtracteurCaracteristiques:

    NOMS = (["stft_" + b for b in AnalyseurTempsFrequence.BANDES]
            + ["cwt_" + b for b in AnalyseurTempsFrequence.BANDES]
            + ["ratio_gamma_alpha", "entropie_spectrale", "aplatissement",
               "variation_amplitude"])

    def __init__(self, fs=256):
        self.tf = AnalyseurTempsFrequence(fs)

    def calculer_entropie_spectrale(self, P):
        dsp = P.mean(axis=1)
        p = dsp / (dsp.sum() + EPS)
        return -np.sum(p * np.log2(p + EPS))

    def extraire_un_segment(self, x):
        f_stft, _, P_stft = self.tf.calculer_stft(x)
        f_cwt, P_cwt = self.tf.calculer_cwt(x)

        e_stft = self.tf.energie_par_bande(f_stft, P_stft)
        e_cwt = self.tf.energie_par_bande(f_cwt, P_cwt)

        ratio_gamma_alpha = e_stft["gamma"] / (e_stft["alpha"] + EPS)

        moitie = len(x) // 2
        variation = x[moitie:].std() / (x[:moitie].std() + 1e-8)  

        aplatissement = ((x - x.mean()) ** 4).mean() / (x.var() ** 2 + EPS)

        return np.array(
            [e_stft[b] for b in AnalyseurTempsFrequence.BANDES]
            + [e_cwt[b] for b in AnalyseurTempsFrequence.BANDES]
            + [ratio_gamma_alpha, self.calculer_entropie_spectrale(P_stft),
               aplatissement, variation]
        )

    def extraire_tous(self, X):
        return np.array([self.extraire_un_segment(seg) for seg in X])

    @classmethod
    def index_de(cls, nom):
        return cls.NOMS.index(nom)

    @classmethod
    def controler_descripteurs(cls, F, seuil_variance=1e-10):
      
        variances = F.var(axis=0)
        morts = [cls.NOMS[i] for i, v in enumerate(variances) if v < seuil_variance]
        if morts:
            print(f"  [ALERTE] descripteur(s) sans information : {morts}")
        else:
            print("  [OK] tous les descripteurs ont une variance non nulle")
        return morts


def pipeline_svm(seed=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        # Jeu desequilibre (150 vs 60) donc ponderation inverse des effectifs.
        ("svm", SVC(kernel="rbf", probability=True,
                    class_weight="balanced", random_state=seed)),
    ])


GRILLE_HYPERPARAMETRES = {"svm__C": [0.1, 1, 10, 100],
                          "svm__gamma": ["scale", "auto", 0.01, 0.1]}


def modele_avec_recherche(seed=42):
    """SVM + GridSearchCV interne : c'est cet objet complet que l'on valide."""
    return GridSearchCV(pipeline_svm(seed), GRILLE_HYPERPARAMETRES,
                        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                        scoring="f1", n_jobs=-1)


def validation_imbriquee(X, y, seed=42):
    """Boucle externe d'evaluation, boucle interne de selection : score non biaise."""
    externe = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_score(modele_avec_recherche(seed), X, y, cv=externe, scoring="f1")


if __name__ == "__main__":
    SEED = 42
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    X_brut, y = GenerateurDonneesEEG(graine_aleatoire=SEED).construire_jeu_de_donnees()
    X_propre = PretraiteurEEG().pretraiter_tous_segments(X_brut)
    F = ExtracteurCaracteristiques().extraire_tous(X_propre)

    print(f"Matrice de features : {F.shape}   (segments x descripteurs)")
    print(f"Classes : {np.bincount(y)}  -> {100 * y.mean():.0f} % de crises")
    ExtracteurCaracteristiques.controler_descripteurs(F)
    print()

    scores = validation_imbriquee(F, y, SEED)
    print(f"F1 (CV imbriquee)          : {scores.mean():.4f} +/- {scores.std():.4f}")

    for nom, strategie in [("classe majoritaire", "most_frequent"),
                           ("tirage stratifie", "stratified")]:
        naif = cross_val_score(DummyClassifier(strategy=strategie, random_state=SEED),
                               F, y, cv=cv, scoring="f1")
        print(f"F1 (naif, {nom:<18s}) : {naif.mean():.4f}")

    print()
    for nom in ("variation_amplitude", "aplatissement", "ratio_gamma_alpha", "cwt_gamma"):
        j = ExtracteurCaracteristiques.index_de(nom)
        s = cross_val_score(pipeline_svm(SEED), F[:, [j]], y, cv=cv, scoring="f1")
        print(f"F1 ({nom:<20s} SEUL) : {s.mean():.4f}")

    y_pred = cross_val_predict(modele_avec_recherche(SEED), F, y, cv=cv)
    print("\nRapport de classification (predictions hors-echantillon) :")
    print(classification_report(y, y_pred, target_names=["Normal", "Crise"]))