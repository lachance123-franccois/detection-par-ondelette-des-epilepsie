# EEG Seizure Detection
## Détection automatique de crises épileptiques par analyse temps-fréquence et apprentissage automatique

> Pipeline complet : signal EEG brut → prétraitement → analyse STFT/CWT → features spectrales → SVM → décision ictal / interictal

```
x(t) brut → Filtre Butterworth + Notch → STFT / CWT → Feature Vector (16 dim) → SVM RBF → {Crise, Normal}
```

---

## Table des matières

1. [Contexte médical et physique](#1-contexte-médical-et-physique)
2. [Modèle biophysique du signal EEG](#2-modèle-biophysique-du-signal-eeg)
3. [Fondements mathématiques de l'analyse temps-fréquence](#3-fondements-mathématiques-de-lanalyse-temps-fréquence)
4. [Prétraitement du signal](#4-prétraitement-du-signal)
5. [Extraction de features spectrales](#5-extraction-de-features-spectrales)
6. [Classification par SVM RBF](#6-classification-par-svm-rbf)
7. [Validation et métriques cliniques](#7-validation-et-métriques-cliniques)
8. [Installation et utilisation](#8-installation-et-utilisation)
9. [Structure du projet](#9-structure-du-projet)
10. [Dataset : CHB-MIT](#10-dataset--chb-mit)
11. [Références bibliographiques](#11-références-bibliographiques)

---

## 1. Contexte médical et physique

### 1.1 L'épilepsie comme problème de dynamique neuronale

L'épilepsie touche environ **50 millions de personnes** dans le monde (OMS, 2022). Une crise épileptique est le résultat d'une **hypersynchronisation pathologique** d'un grand ensemble de neurones corticaux. En temps normal, les réseaux neuronaux maintiennent un équilibre entre excitation (glutamate) et inhibition (GABA). Lors d'une crise, cet équilibre est rompu : une population de neurones se met à décharger de manière synchronisée et répétitive, envahissant progressivement le cortex.

L'EEG (*électroencéphalogramme*) mesure, via des électrodes posées sur le scalp, les **fluctuations de potentiel électrique** générées par la sommation des post-potentiels synaptiques (PSP) de millions de neurones pyramidaux des couches corticales III et V. C'est la seule modalité d'imagerie capable de capturer la dynamique milliseconde de ces phénomènes.

### 1.2 États cliniques et signatures EEG

On distingue deux états fondamentaux dans le contexte de l'épilepsie :

| État | Terminologie | Description clinique | Signature EEG |
|------|-------------|---------------------|---------------|
| **Normal** | Interictal | Activité cérébrale de repos ou tâche cognitive | Rythmes organisés α, β, oscillations à basse amplitude |
| **Crise** | Ictal | Décharge paroxystique hypersynchrone | Complexes Pointe-Onde (Spike-and-Wave), amplitude ↑↑, fréquence ↑↑ |
| **Post-crise** | Postictal | Dépression corticale après la crise | Ondes lentes δ dominantes, atténuation généralisée |

Les **complexes Pointe-Onde** (*Spike-and-Wave*) sont la signature la plus caractéristique : une pointe (durée < 70 ms, amplitude 200–1000 µV) suivie d'une onde lente (durée 200–500 ms). Ils se répètent à 3 Hz dans l'épilepsie-absences de l'enfant, et à 1–2.5 Hz dans les épilepsies focales complexes.

### 1.3 Les bandes de fréquences cérébrales

Le signal EEG est conventionnellement décomposé en bandes de fréquences, chacune associée à un état cognitif ou pathologique précis :

| Bande | Plage (Hz) | État associé | Rôle dans la détection |
|-------|-----------|-------------|----------------------|
| **Delta** (δ) | 0.5 – 4 | Sommeil profond, lésion cérébrale | Augmente en postictal |
| **Thêta** (θ) | 4 – 8 | Somnolence, mémoire | Précède parfois la crise |
| **Alpha** (α) | 8 – 13 | Éveil calme, yeux fermés | Diminue fortement en ictal |
| **Bêta** (β) | 13 – 30 | Activité cognitive, motrice | Activité de fond normale |
| **Gamma** (γ) | 30 – 70 | Hypersynchronisation, traitement sensoriel | **Marqueur majeur de crise** ↑↑ |

Le ratio **γ/α** est l'indicateur le plus discriminant dans ce projet : il est typiquement inférieur à 1 en interictal et monte à 5–20 en phase ictale.

---

## 2. Modèle biophysique du signal EEG

### 2.1 Génération du signal

Un signal EEG mesuré à l'électrode `k` peut être modélisé comme la superposition linéaire des contributions de `N` sources dipolaires corticales, filtrées par la conductivité des tissus (crâne, LCR, cuir chevelu) :

```
x_k(t) = Σ_{n=1}^{N}  a_{kn} · s_n(t)  +  n_k(t)
```

où `a_{kn}` est le coefficient de mélange (lié à la géométrie source-électrode et aux propriétés conductrices des tissus), `s_n(t)` est l'activité de la n-ième source neuronale, et `n_k(t)` est le bruit de mesure (thermique + artéfacts).

### 2.2 Modèle de la dynamique ictale

Pendant une crise, la dynamique d'un ensemble neuronal est souvent modélisée par un oscillateur de van der Pol modulé en fréquence :

```
ẍ − μ(1 − x²)ẋ + ω²(t)·x = F(t)
```

où `ω(t)` est la fréquence instantanée (qui augmente au début de la crise — phénomène de *frequency evolution*) et `μ` contrôle l'amplitude des oscillations non-linéaires. Ce modèle justifie pourquoi la **non-stationnarité** est intrinsèque au signal ictal : `ω(t)` varie dans le temps, rendant la FFT classique inadaptée.

### 2.3 Rapport Signal-sur-Bruit (SNR) et défis

Le SNR d'un EEG de scalp est typiquement **−10 à +10 dB** : les sources d'artéfacts (clignements oculaires : 100–200 µV, mouvements musculaires : 50–1000 µV) peuvent être bien supérieures en amplitude aux signaux cérébraux d'intérêt (20–100 µV en interictal). Ce constat motive l'étape de filtrage décrite en section 4.

---

## 3. Fondements mathématiques de l'analyse temps-fréquence

### 3.1 Pourquoi la FFT est insuffisante

La Transformée de Fourier classique suppose la **stationnarité du signal** — hypothèse radicalement violée par l'EEG. Formellement, la FFT calcule :

```
X(f) = ∫_{-∞}^{+∞} x(t) · e^{-j2πft} dt
```

Cette représentation donne le **contenu fréquentiel global** sur toute la durée du signal, mais efface toute information temporelle : si une bouffée gamma n'apparaît que pendant 500 ms sur une fenêtre de 10 s, son amplitude dans `|X(f)|` sera diluée par un facteur 20. Elle sera donc invisible.

Le problème fondamental est lié au **principe d'incertitude de Gabor** (1946) :

```
Δt · Δf ≥ 1/(4π)
```

Il est **impossible** d'avoir simultanément une résolution temporelle et fréquentielle arbitrairement fine. Toute méthode temps-fréquence est un compromis entre ces deux résolutions.

---

### 3.2 Short-Time Fourier Transform (STFT)

#### Définition

La STFT résout le problème de stationnarité en appliquant la FFT sur des **fenêtres temporelles glissantes** de durée finie `T_w` :

```
STFT{x(t)}(τ, f) = ∫_{-∞}^{+∞} x(t) · w(t − τ) · e^{-j2πft} dt
```

où `w(t − τ)` est une fonction de fenêtrage centrée sur l'instant `τ`, et le spectrogramme (représentation visuelle utilisée dans ce projet) est le carré du module :

```
S(τ, f) = |STFT{x(t)}(τ, f)|²    [V²/Hz]
```

#### Discrétisation

En pratique, pour un signal numérique `x[n]` échantillonné à `fs` Hz, avec une fenêtre de `N_w` points et un pas de `H` points (hop size) :

```
STFT[m, k] = Σ_{n=0}^{N_w − 1}  x[n + mH] · w[n] · e^{-j2πkn/N_w}
```

La résolution fréquentielle est `Δf = fs / N_w` et la résolution temporelle est `Δt = H / fs`. Il y a un compromis direct : augmenter `N_w` améliore `Δf` mais dégrade `Δt`.

#### Fenêtre de Hann

La fenêtre de Hann (ou Hanning) est le choix standard pour l'EEG car elle réduit la fuite spectrale (*spectral leakage*) à −31.5 dB :

```
w[n] = 0.5 · (1 − cos(2π·n / (N_w − 1)))    n = 0, ..., N_w − 1
```

Un fenêtrage Hann avec 50% d'overlap garantit la condition de reconstruction parfaite (Constant Overlap-Add, COLA), utile pour la synthèse.

#### Limitation de la STFT : résolution fixe

La STFT a une résolution temps-fréquence **uniforme** sur tout le plan (τ, f). Cela est sous-optimal pour l'EEG où les basses fréquences (δ, θ) évoluent lentement — nécessitant une bonne résolution fréquentielle — tandis que les hautes fréquences (γ) varient rapidement — nécessitant une bonne résolution temporelle. Cette limitation motive la CWT.

---

### 3.3 Transformée en Ondelettes Continues (CWT)

#### Principe : analyse multi-résolution

La CWT décompose le signal en une famille d'**ondelettes filles** obtenues par dilatation et translation d'une **ondelette mère** `ψ(t)` :

```
CWT{x(t)}(a, b) = (1/√|a|) · ∫_{-∞}^{+∞} x(t) · ψ*((t − b)/a) dt
```

où `a > 0` est le **facteur d'échelle** (inverse de la fréquence) et `b` est la **translation temporelle**. Le facteur `1/√|a|` assure la conservation de l'énergie entre les échelles.

La relation entre échelle et fréquence centrale dépend de l'ondelette choisie :

```
f = f_c / (a · Δt)
```

où `f_c` est la fréquence centrale de l'ondelette mère et `Δt = 1/fs` la période d'échantillonnage.

#### Résolution adaptative

La résolution temps-fréquence de la CWT est **adaptative** :

```
Haute fréquence (a petit)  : Δt petit (bonne résolution temporelle), Δf grand
Basse fréquence (a grand)  : Δt grand (bonne résolution fréquentielle), Δf petit
```

Ce comportement est exactement ce que l'on souhaite pour l'EEG : précision temporelle pour les transitoires gamma, précision fréquentielle pour les rythmes alpha/thêta lents.

#### L'ondelette de Morlet complexe

L'ondelette de Morlet est le choix quasi-universel en neurosciences cognitives (Tallon-Baudry & Bertrand, 1999) car sa forme — une sinusoïde modulée par une gaussienne — est physiquement similaire aux bouffées oscillatoires neuronales :

```
ψ(t) = (1 / √(π·f_b)) · e^{j2π·f_c·t} · e^{−t² / f_b}
```

où `f_c` est la fréquence centrale (ici 1 Hz) et `f_b` est le paramètre de largeur de bande (contrôle l'étalement temporel de l'enveloppe gaussienne). Sa transformée de Fourier est :

```
Ψ(f) = e^{−π² · f_b · (f − f_c)²}
```

C'est une gaussienne dans le domaine fréquentiel, centrée sur `f_c`, ce qui lui confère d'excellentes propriétés de localisation spectrale.

**Propriété d'admissibilité :** Pour qu'une ondelette mère soit valide, elle doit satisfaire la condition d'admissibilité (moyenne nulle) :

```
∫_{-∞}^{+∞} ψ(t) dt = 0    ↔    Ψ(0) = 0
```

L'ondelette de Morlet complexe vérifie cette condition approximativement pour `f_c / σ_f > 5` (ratio d'onde), ce qui est la configuration utilisée dans ce projet.

#### Scalogramme

La représentation visuelle équivalente au spectrogramme STFT est le **scalogramme** :

```
SC(a, b) = |CWT{x(t)}(a, b)|²    [proportionnel à V²/Hz]
```

Il est la représentation temps-échelle de la densité d'énergie du signal.

---

### 3.4 Comparaison STFT vs CWT pour l'EEG

| Critère | STFT | CWT (Morlet) |
|---------|------|-------------|
| Résolution temporelle | Uniforme : `Δt = H/fs` | Adaptative : meilleure à haute `f` |
| Résolution fréquentielle | Uniforme : `Δf = fs/N_w` | Adaptative : meilleure à basse `f` |
| Hypothèse | Quasi-stationnarité par fenêtre | Aucune stationnarité requise |
| Complexité | O(N log N) par fenêtre | O(N·N_scales) |
| Adapté aux transitoires | Moyen | Excellent |
| Adapté aux rythmes lents | Bon | Excellent |
| Usage dans ce projet | Spectrogramme de surveillance | Features multi-résolution |

---

## 4. Prétraitement du signal

### 4.1 Architecture de la chaîne de filtrage

```
x_raw(t)  →  [Filtre Butterworth 0.5–70 Hz]  →  [Filtre Notch 50 Hz]  →  x_clean(t)
```

### 4.2 Filtre passe-bande Butterworth d'ordre 4

Le filtre de Butterworth est dit **maximalement plat** dans la bande passante : sa réponse fréquentielle ne présente aucune ondulation (contrairement à Chebyshev ou elliptique). Sa fonction de transfert en z-plan est :

```
|H(f)|² = 1 / (1 + (f/fc)^{2N})
```

Pour `N = 4` (ordre 4), l'atténuation hors-bande est de **−80 dB/décade** (24 dB/octave). La bande passante [0.5, 70] Hz est choisie pour :

- **Borne inférieure 0.5 Hz :** éliminer la dérive de ligne de base (mouvements de tête, respiration) et les artéfacts de transpiration (< 0.1 Hz), tout en conservant les ondes δ lentes (0.5–4 Hz).
- **Borne supérieure 70 Hz :** conserver l'intégralité des bandes d'intérêt (jusqu'à γ) tout en rejetant les artéfacts musculaires haute fréquence (EMG > 80 Hz).

L'implémentation utilise `scipy.signal.filtfilt` (filtrage zéro-phase) pour éviter toute distorsion de phase — critique en EEG car le délai de phase altère les relations de synchronisation inter-électrodes :

```python
def filtrer_signal(self, signal_eeg: np.ndarray) -> np.ndarray:
    """
    Filtrage zéro-phase : applique le filtre en avant puis en arrière.
    La phase est exactement nulle → aucun décalage temporel des événements.
    Coût : doublement de l'ordre effectif (ordre 4 → atténuation d'ordre 8).
    """
    signal_filtre = scipy.signal.filtfilt(self.b_band, self.a_band, signal_eeg)
    return signal_filtre
```

**Justification du filtrage zéro-phase :** `filtfilt` applique le filtre une première fois en avant (introduisant un délai de groupe `τ(f)`), puis une seconde fois en arrière (introduisant `−τ(f)`). Le délai net est nul pour toutes les fréquences. Le prix à payer est un doublement de l'ordre effectif (8 pôles au lieu de 4), soit une atténuation hors-bande de **−160 dB/décade**.

### 4.3 Filtre Notch à 50 Hz

Le réseau électrique européen rayonne à 50 Hz et ses harmoniques (100, 150 Hz...). Ce bruit d'induction électromagnétique peut saturer l'amplificateur EEG. Un filtre coupe-bande IIR à 50 Hz est appliqué :

```python
b_notch, a_notch = scipy.signal.iirnotch(w0=50.0, Q=30.0, fs=fs)
```

Le facteur de qualité `Q = f0 / Δf = 30` donne une largeur de bande à −3 dB de `Δf = 50/30 ≈ 1.67 Hz`, suffisamment étroite pour ne pas perturber les bandes β et γ adjacentes.

### 4.4 Normalisation Z-score

Avant l'extraction de features, chaque segment est normalisé pour rendre les features comparables entre sujets et sessions :

```
x_norm[n] = (x_clean[n] − μ_x) / σ_x
```

Cette étape est essentielle car les amplitudes EEG varient d'un facteur 10 entre sujets selon l'épaisseur du crâne et la qualité du contact électrode-scalp.

---

## 5. Extraction de features spectrales

### 5.1 Segmentation et fenêtrage

Le signal continu est découpé en segments de durée fixe `T_w = 4 s` avec un recouvrement de 50% :

```
Nombre de segments ≈ (T_total − T_w) / (T_w × 0.5) + 1
```

Le choix `T_w = 4 s` est un compromis entre la stationnarité locale (l'EEG est quasi-stationnaire sur 2–5 s) et la résolution fréquentielle : à `fs = 256 Hz`, une fenêtre de 4 s donne `N = 1024` points et `Δf = 0.25 Hz`, permettant de résoudre les bandes δ (0.5–4 Hz) avec 14 bins.

### 5.2 Énergie par bande — Méthode de Welch

L'énergie dans chaque bande est estimée par intégration de la PSD de Welch :

```
E_band = ∫_{f_low}^{f_high} S_xx(f) df ≈ Σ_{k : f_low ≤ f_k < f_high} S_xx[k] · Δf    [V²]
```

La PSD de Welch est préférée au périodogramme brut car elle réduit la variance de l'estimateur d'un facteur M (nombre de sous-segments) tout en conservant une résolution acceptable :

```python
f_welch, psd = scipy.signal.welch(x_norm, fs=fs, window='hann',
                                   nperseg=256, noverlap=128)
```

Les 5 features d'énergie par bande sont : `E_delta`, `E_theta`, `E_alpha`, `E_beta`, `E_gamma`.

### 5.3 Ratio γ/α — marqueur principal de crise

```
R_gamma_alpha = E_gamma / (E_alpha + ε)    avec ε = 1e-12 pour éviter la division par zéro
```

Ce ratio est le **marqueur le plus discriminant** dans la littérature (Shoeb, 2009). En interictal, le spectre EEG est dominé par α (8–13 Hz) et le ratio est < 1. En ictal, l'énergie gamma explose par hypersynchronisation et le ratio monte à 5–20.

### 5.4 Entropie spectrale

L'entropie spectrale mesure le degré de désordre du spectre de puissance normalisé. Elle est définie comme l'entropie de Shannon appliquée à la distribution spectrale normalisée `p(f) = S_xx(f) / Σ S_xx(f)` :

```
!H_spec = −Σ_k p(f_k) · log₂(p(f_k))    [bits]
```

**Interprétation physique :** Un spectre plat (bruit blanc) a une entropie maximale `H_max = log₂(N_bins)`. Un signal ictal, bien que de grande amplitude, est **spectralement concentré** autour des fréquences de décharge synchrone → l'entropie **diminue** lors d'une crise. C'est un indicateur contre-intuitif mais robuste.

### 5.5 Features CWT multi-résolution

Pour chaque échelle `a` correspondant aux bandes δ, θ, α, β, γ, on calcule l'énergie dans le scalogramme :

```
E_CWT(f_band) = Σ_b |CWT{x}(a_band, b)|² · δb
```

Ces 5 features CWT complètent les 5 features Welch avec une meilleure résolution temporelle pour les transitoires rapides (spike gamma).

### 5.6 Features statistiques temporelles complémentaires

| Feature | Formule | Intérêt pour l'EEG ictal |
|---------|---------|--------------------------|
| **RMS** | `√(mean(x²))` | Amplitude globale ↑↑ en ictal |
| **Kurtosis** | `E[(x−μ)⁴]/σ⁴` | Détecte les pointes (spikes) |
| **Zero Crossing Rate** | `Σ |sgn(x[n+1])−sgn(x[n])| / (2N)` | Fréquence apparente ↑ en ictal |

### 5.7 Vecteur de features complet

Le vecteur final de **16 features** est :

```
f = [E_δ, E_θ, E_α, E_β, E_γ,          5 énergies Welch
     R_γ/α,                              1 ratio spectral
     H_spec,                             1 entropie spectrale
     E_CWT_δ, E_CWT_θ, E_CWT_α,
     E_CWT_β, E_CWT_γ,                  5 énergies CWT
     RMS, kurtosis, ZCR]                3 statistiques temporelles
```

---

## 6. Classification par SVM RBF

### 6.1 Formulation du problème

La classification est un problème binaire : `y ∈ {−1 (interictal), +1 (ictal)}`. Le vecteur de features `f ∈ ℝ¹⁶` est d'abord normalisé par `StandardScaler` (moyenne nulle, variance unitaire par feature) pour que le noyau RBF ne soit pas biaisé par les différences d'échelle entre features.

### 6.2 Le noyau RBF — justification mathématique

Le noyau RBF (*Radial Basis Function*) ou noyau gaussien est défini par :

```
K(x, x') = exp(−γ · ‖x − x'‖²)    γ > 0
```

Par le théorème de Mercer, ce noyau est défini positif et correspond à un produit scalaire dans un espace de Hilbert de dimension infinie `H` :

```
K(x, x') = ⟨φ(x), φ(x')⟩_H
```

Le développement en série de Taylor de l'exponentielle montre que le mapping implicite `φ` contient des monômes de **tous les degrés** : le noyau RBF peut donc séparer des classes non-linéairement séparables dans `ℝ¹⁶`.

**Interprétation :** `γ` contrôle la portée du noyau. Pour `γ` grand, `K(x, x') ≈ 0` dès que `x` et `x'` sont légèrement différents → frontière de décision très irrégulière (overfitting). Pour `γ` petit, le noyau est large → frontière lisse (underfitting). Valeur optimale trouvée par GridSearchCV.

### 6.3 Problème d'optimisation SVM

Le SVM cherche l'hyperplan de séparation de marge maximale dans l'espace `H` :

```
min_{w, b, ξ}  ½‖w‖² + C · Σᵢ ξᵢ

sous contraintes :
  yᵢ (⟨w, φ(xᵢ)⟩ + b) ≥ 1 − ξᵢ    ∀i
  ξᵢ ≥ 0                              ∀i
```

La formulation duale (utilisée en pratique via les multiplicateurs de Lagrange `αᵢ`) est :

```
max_{α}  Σᵢ αᵢ − ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ)

sous :   0 ≤ αᵢ ≤ C    et    Σᵢ αᵢ yᵢ = 0
```

La décision finale pour un nouveau segment `x_test` est :

```
ŷ = sgn(Σᵢ αᵢ yᵢ K(xᵢ, x_test) + b)
```

Seuls les **vecteurs supports** (points avec `αᵢ > 0`, situés sur ou dans la marge) contribuent à la décision.

### 6.4 Optimisation des hyperparamètres

`GridSearchCV` avec validation croisée stratifiée 5-fold explore la grille :

```python
param_grid = {
    'C':     [0.1, 1, 10, 100],       # compromis marge / erreurs
    'gamma': [0.001, 0.01, 0.1, 1],   # portée du noyau RBF
}
```

La métrique d'optimisation est le **F1-score** (et non l'accuracy) car les classes sont déséquilibrées : en EEG clinique continu, les segments ictaux représentent souvent moins de 5% du total.

---

## 7. Validation et métriques cliniques

### 7.1 Métriques adaptées au contexte médical

En contexte clinique, les conséquences d'une **fausse alarme** (segment normal classé comme crise) et d'un **manqué** (crise non détectée) sont asymétriques. On utilise donc :

```
Sensibilité (Recall)    = TP / (TP + FN)    minimiser les crises manquées
Spécificité             = TN / (TN + FP)    minimiser les fausses alarmes
F1-score                = 2·P·R / (P+R)     compromis global
AUC-ROC                                     performance indépendante du seuil
```

### 7.2 Objectifs de performance

D'après la littérature clinique (Shoeb, 2009 ; Subasi, 2007), les seuils considérés comme cliniquement acceptables sont :

| Métrique | Seuil clinique | Cible du projet |
|----------|---------------|----------------|
| Sensibilité | > 90 % | > 92 % |
| Spécificité | > 95 % | > 96 % |
| Latence de détection | < 10 s | < 5 s |
| Taux de fausses alarmes | < 1/h | < 0.5/h |

### 7.3 Validation croisée Leave-One-Patient-Out (LOPO)

Pour éviter le **data leakage** inter-patient (un modèle entraîné sur des segments du patient A ne doit pas être évalué sur des segments du même patient), la validation LOPO est recommandée : entraîner sur N−1 patients, évaluer sur le patient restant, répéter N fois.

---

## 8. Installation et utilisation

### 8.1 Installation

```bash
git clone https://github.com/ton-user/detection-par-ondelette-des-epilepsie.git
cd detection-par-ondelette-des-epilepsie
```

**Dépendances :**

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
scikit-learn>=1.3
PyWavelets>=1.4
mne>=1.5          # lecture des fichiers EDF/EDF+ du dataset CHB-MIT
```

### 8.2 Utilisation rapide — données synthétiques

```python
from pipeline import EEGPipeline

# Instanciation avec paramètres par défaut (fs=256 Hz, T_w=4s, overlap=0.5)
pipe = EEGPipeline(fs=256, window_sec=4.0, overlap=0.5)

# Prédiction sur un fichier EDF (format CHB-MIT)
result = pipe.predict_from_edf("chb01_03.edf", channel="FP1-F7")
print(result)
# {'label': 'ictal', 'confidence': 0.97, 'features': {...}, 'onset_sec': 2996.0}

# Simulation : génère un signal synthétique et trace le spectrogramme
pipe.demo_synthetic(duration_sec=30, show_wavelet=True)
```

### 8.3 Visualisation

```python
# Spectrogramme STFT
pipe.plot_spectrogram(x, title="EEG Ictal — Patient chb01")

# Scalogramme CWT (ondelette de Morlet)
pipe.plot_scalogram(x, scales=np.arange(1, 128))

# Comparaison interictal vs ictal
pipe.plot_comparison(x_normal, x_seizure)
```



## 9. Dataset : CHB-MIT

Le **CHB-MIT Scalp EEG Database** (Shoeb & Guttag, 2009) est la référence académique mondiale pour la détection de crises épileptiques. Il est disponible librement sur PhysioNet :

- **23 patients** pédiatriques (5–22 ans), principalement résistants au traitement
- **916 heures** d'enregistrement EEG continu
- **198 crises** annotées manuellement par des neurologues
- Format : **EDF+** (European Data Format), 23 canaux, 256 Hz
- Lien : https://physionet.org/content/chbmit/1.0.0/

**Téléchargement :**

```bash
# Avec wget (recommandé)
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/

# Ou avec le client PhysioNet
pip install wfdb
python -c "import wfdb; wfdb.dl_database('chbmit', './data/chb-mit/')"
```

**Lecture avec MNE-Python :**

```python
import mne
raw = mne.io.read_raw_edf("data/chb-mit/chb01/chb01_03.edf", preload=True)
x = raw.get_data(picks=["FP1-F7"])[0]  # canal FP1-F7, signal 1D
fs = int(raw.info['sfreq'])             # 256 Hz
```

---

## 11. Références bibliographiques

| # | Référence |
|---|-----------|
| [1] | Shoeb, A.H. & Guttag, J.V. (2010). Application of Machine Learning to Epileptic Seizure Detection. *Proc. 27th ICML*, 975–982. https://people.csail.mit.edu/jguttag/papers/shoeb-icml10.pdf |
| [2] | Shoeb, A.H. (2009). *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment*. PhD Thesis, MIT. https://dspace.mit.edu/handle/1721.1/54669 |
| [3] | Mallat, S. (1999). *A Wavelet Tour of Signal Processing* (2nd ed.). Academic Press. https://doi.org/10.1016/B978-0-12-466606-1.X5000-4 |
| [4] | Buzsáki, G. (2006). *Rhythms of the Brain*. Oxford University Press. https://doi.org/10.1093/acprof:oso/9780195301069.001.0001 |
| [5] | Gabor, D. (1946). Theory of Communication. *Journal of the IEE*, 93(26), 429–457. https://doi.org/10.1049/ji-3-2.1946.0074 |
| [6] | Tallon-Baudry, C. & Bertrand, O. (1999). Oscillatory gamma activity in humans and its role in object representation. *Trends in Cognitive Sciences*, 3(4), 151–162. https://doi.org/10.1016/S1364-6613(99)01299-1 |
| [7] | Cortes, C. & Vapnik, V. (1995). Support-Vector Networks. *Machine Learning*, 20(3), 273–297. https://doi.org/10.1007/BF00994018 |
| [8] | Subasi, A. (2007). EEG signal classification using wavelet feature extraction and a mixture of expert model. *Expert Systems with Applications*, 32(4), 1084–1093. https://doi.org/10.1016/j.eswa.2006.02.005 |
| [9] | Goldberger, A.L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220. https://doi.org/10.1161/01.CIR.101.23.e215 |
| [10] | Welch, P.D. (1967). The Use of Fast Fourier Transform for the Estimation of Power Spectra. *IEEE Trans. Audio Electroacoust.*, 15(2), 70–73. https://doi.org/10.1109/TAU.1967.1161901 |
| [11] | Niedermeyer, E. & da Silva, F.L. (2004). *Electroencephalography: Basic Principles, Clinical Applications, and Related Fields* (5th ed.). Lippincott Williams & Wilkins. |
| [12] | Tzallas, A.T., Tsipouras, M.G. & Fotiadis, D.I. (2009). Epileptic Seizure Detection in EEGs Using Time–Frequency Analysis. *IEEE Trans. Information Technology in Biomedicine*, 13(5), 703–710. https://doi.org/10.1109/TITB.2009.2017939 |

---