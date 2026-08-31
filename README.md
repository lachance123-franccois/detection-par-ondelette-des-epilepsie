# EEG Seizure Detection

## Détection de crises épileptiques par analyse temps-fréquence et apprentissage automatique

> Pipeline complet : signal EEG **synthétique** → prétraitement → analyse STFT/CWT → 14 descripteurs → SVM RBF → décision ictal / interictal

```
x(t) → Butterworth 0.5–70 Hz + Notch 50 Hz → z-score → STFT + CWT → f ∈ ℝ¹⁴ → SVM RBF → {Crise, Normal}
```

> **Statut du projet.** Le code de ce dépôt fonctionne intégralement sur des signaux **générés analytiquement** (`GenerateurDonneesEEG`). Aucun EEG réel n'est lu : l'intégration du dataset CHB-MIT décrite en section 11 est une piste d'extension, pas une fonctionnalité existante. Les scores rapportés en section 8 (F1 = 1.00) mesurent la cohérence du pipeline sur un problème séparable par construction — **ils ne constituent pas une validation clinique**.

---

## Table des matières

1. [Contexte médical et physique](#1-contexte-médical-et-physique)
2. [Modèle biophysique du signal EEG](#2-modèle-biophysique-du-signal-eeg)
3. [Fondements mathématiques de l'analyse temps-fréquence](#3-fondements-mathématiques-de-lanalyse-temps-fréquence)
4. [Génération des données synthétiques](#4-génération-des-données-synthétiques)
5. [Prétraitement du signal](#5-prétraitement-du-signal)
6. [Extraction des descripteurs](#6-extraction-des-descripteurs)
7. [Classification par SVM RBF](#7-classification-par-svm-rbf)
8. [Validation et résultats](#8-validation-et-résultats)
9. [Installation et utilisation](#9-installation-et-utilisation)
10. [Structure du projet et limites](#10-structure-du-projet-et-limites)
11. [Extension : dataset CHB-MIT](#11-extension--dataset-chb-mit)
12. [Références bibliographiques](#12-références-bibliographiques)

---

## 1. Contexte médical et physique

### 1.1 L'épilepsie comme problème de dynamique neuronale

L'épilepsie touche environ **50 millions de personnes** dans le monde (OMS, 2022). Une crise épileptique est le résultat d'une **hypersynchronisation pathologique** d'un grand ensemble de neurones corticaux. En temps normal, les réseaux neuronaux maintiennent un équilibre entre excitation (glutamate) et inhibition (GABA). Lors d'une crise, cet équilibre est rompu : une population de neurones se met à décharger de manière synchronisée et répétitive, envahissant progressivement le cortex.

L'EEG (*électroencéphalogramme*) mesure, via des électrodes posées sur le scalp, les **fluctuations de potentiel électrique** générées par la sommation des post-potentiels synaptiques (PSP) de millions de neurones pyramidaux des couches corticales III et V. C'est la seule modalité d'imagerie capable de capturer la dynamique milliseconde de ces phénomènes.

### 1.2 États cliniques et signatures EEG

| État | Terminologie | Description clinique | Signature EEG |
|------|-------------|---------------------|---------------|
| **Normal** | Interictal | Activité cérébrale de repos ou tâche cognitive | Rythmes organisés α, β, oscillations à basse amplitude |
| **Crise** | Ictal | Décharge paroxystique hypersynchrone | Complexes Pointe-Onde (Spike-and-Wave), amplitude ↑↑, fréquence ↑↑ |
| **Post-crise** | Postictal | Dépression corticale après la crise | Ondes lentes δ dominantes, atténuation généralisée |

Seuls les deux premiers états sont modélisés dans ce projet : le classifieur est **binaire** (interictal vs ictal), l'état postictal n'est ni généré ni détecté.

Les **complexes Pointe-Onde** (*Spike-and-Wave*) sont la signature la plus caractéristique : une pointe (durée < 70 ms, amplitude 200–1000 µV) suivie d'une onde lente (durée 200–500 ms). Ils se répètent à 3 Hz dans l'épilepsie-absences de l'enfant, et à 1–2.5 Hz dans les épilepsies focales complexes.

### 1.3 Les bandes de fréquences cérébrales

| Bande | Plage (Hz) | État associé | Rôle dans la détection |
|-------|-----------|-------------|----------------------|
| **Delta** (δ) | 0.5 – 4 | Sommeil profond, lésion cérébrale | Augmente en postictal |
| **Thêta** (θ) | 4 – 8 | Somnolence, mémoire | Précède parfois la crise |
| **Alpha** (α) | 8 – 13 | Éveil calme, yeux fermés | Diminue fortement en ictal |
| **Bêta** (β) | 13 – 30 | Activité cognitive, motrice | Activité de fond normale |
| **Gamma** (γ) | 30 – 70 | Hypersynchronisation, traitement sensoriel | **Marqueur majeur de crise** ↑↑ |

Ces bornes sont exactement celles codées dans `AnalyseurTempsFrequence.BANDES`. Les intervalles y sont **semi-ouverts** `[lo, hi)` — sauf le dernier, `[30, 70]` — afin qu'aucun bin de fréquence ne soit comptabilisé dans deux bandes voisines.

---

## 2. Modèle biophysique du signal EEG

### 2.1 Génération du signal

Un signal EEG mesuré à l'électrode `k` peut être modélisé comme la superposition linéaire des contributions de `N` sources dipolaires corticales, filtrées par la conductivité des tissus (crâne, LCR, cuir chevelu) :

```
x_k(t) = Σ_{n=1}^{N}  a_{kn} · s_n(t)  +  n_k(t)
```

où `a_{kn}` est le coefficient de mélange (lié à la géométrie source-électrode et aux propriétés conductrices des tissus), `s_n(t)` est l'activité de la n-ième source neuronale, et `n_k(t)` est le bruit de mesure (thermique + artéfacts).

C'est ce modèle additif que reproduit le générateur synthétique : somme de composantes oscillatoires + bruit gaussien (section 4).

### 2.2 Modèle de la dynamique ictale

Pendant une crise, la dynamique d'un ensemble neuronal est souvent modélisée par un oscillateur de van der Pol modulé en fréquence :

```
ẍ − μ(1 − x²)ẋ + ω²(t)·x = F(t)
```

où `ω(t)` est la fréquence instantanée (qui augmente au début de la crise — phénomène de *frequency evolution*) et `μ` contrôle l'amplitude des oscillations non-linéaires. Ce modèle justifie pourquoi la **non-stationnarité** est intrinsèque au signal ictal : `ω(t)` varie dans le temps, rendant la FFT classique inadaptée.

> Note d'implémentation : cette équation différentielle n'est **pas intégrée numériquement** dans le code. Le générateur en imite phénoménologiquement les effets observables (bouffées gamma, enveloppe d'amplitude croissante, pointe-onde à 3 Hz).

### 2.3 Rapport signal-sur-bruit et défis

Le SNR d'un EEG de scalp est typiquement **−10 à +10 dB** : les sources d'artéfacts (clignements oculaires : 100–200 µV, mouvements musculaires : 50–1000 µV) peuvent être bien supérieures en amplitude aux signaux cérébraux d'intérêt (20–100 µV en interictal). Ce constat motive l'étape de filtrage décrite en section 5.

---

## 3. Fondements mathématiques de l'analyse temps-fréquence

### 3.1 Pourquoi la FFT est insuffisante

La Transformée de Fourier classique suppose la **stationnarité du signal** — hypothèse radicalement violée par l'EEG :

```
X(f) = ∫_{-∞}^{+∞} x(t) · e^{-j2πft} dt
```

Cette représentation donne le **contenu fréquentiel global** sur toute la durée du signal, mais efface toute information temporelle : si une bouffée gamma n'apparaît que pendant 500 ms sur une fenêtre de 10 s, son amplitude dans `|X(f)|` sera diluée par un facteur 20.

Le problème fondamental est lié au **principe d'incertitude de Gabor** (1946) :

```
Δt · Δf ≥ 1/(4π)
```

Il est **impossible** d'avoir simultanément une résolution temporelle et fréquentielle arbitrairement fine. Toute méthode temps-fréquence est un compromis entre ces deux résolutions.

---

### 3.2 Short-Time Fourier Transform (STFT)

#### Définition

```
STFT{x(t)}(τ, f) = ∫_{-∞}^{+∞} x(t) · w(t − τ) · e^{-j2πft} dt
```

où `w(t − τ)` est une fenêtre centrée sur l'instant `τ`. Le spectrogramme utilisé comme source de descripteurs est le carré du module :

```
S(τ, f) = |STFT{x(t)}(τ, f)|²
```

#### Discrétisation et paramètres réellement utilisés

```
STFT[m, k] = Σ_{n=0}^{N_w − 1}  x[n + mH] · w[n] · e^{-j2πkn/N_w}
```

La résolution fréquentielle est `Δf = fs / N_w`, la résolution temporelle `Δt = H / fs`.

| Paramètre | Valeur dans `AnalyseurTempsFrequence` | Conséquence |
|-----------|---------------------------------------|-------------|
| `fs` | 256 Hz | — |
| `nperseg` (`N_w`) | 256 points (1 s) | `Δf = 1 Hz`, 129 bins de 0 à 128 Hz |
| `noverlap` | 128 points (50 %) | `H = 128`, soit `Δt = 0.5 s` |
| Fenêtre | Hann | Fuite spectrale ≈ −31.5 dB |

Avec `Δf = 1 Hz`, la bande δ (0.5–4 Hz) est couverte par 3 bins (1, 2, 3 Hz) et la bande γ par 41 bins. C'est le minimum acceptable pour les bandes basses ; descendre `nperseg` à 128 (`Δf = 2 Hz`) ne laisserait qu'un seul bin dans δ.

#### Fenêtre de Hann

```
w[n] = 0.5 · (1 − cos(2π·n / (N_w − 1)))    n = 0, ..., N_w − 1
```

Un fenêtrage Hann avec 50 % de recouvrement satisfait la condition COLA (*Constant Overlap-Add*), ce qui rendrait la STFT inversible — propriété non exploitée ici, puisque seul le spectrogramme est utilisé.

#### Limitation : résolution fixe

La STFT a une résolution temps-fréquence **uniforme** sur tout le plan (τ, f). C'est sous-optimal pour l'EEG, où les basses fréquences (δ, θ) demandent une bonne résolution fréquentielle tandis que les transitoires gamma demandent une bonne résolution temporelle. Cette limitation motive la CWT.

---

### 3.3 Transformée en ondelettes continues (CWT)

> **Précision de vocabulaire.** Le code utilise `pywt.cwt` — une transformée en ondelettes **continue** avec ondelette de Morlet complexe. Ce n'est **pas** une DWT (décomposition dyadique de Mallat en approximations/détails). Le commentaire correspondant est explicite dans `calculer_cwt`.

```
CWT{x(t)}(a, b) = (1/√|a|) · ∫_{-∞}^{+∞} x(t) · ψ*((t − b)/a) dt
```

où `a > 0` est le **facteur d'échelle** et `b` la **translation temporelle**. Le facteur `1/√|a|` assure la conservation de l'énergie entre échelles.

La relation échelle ↔ fréquence dépend de l'ondelette :

```
f = f_c / (a · Δt)        avec Δt = 1/fs
```

Dans le code, cette conversion est faite dans le bon sens par `pywt.frequency2scale` : on **choisit d'abord** 64 fréquences logarithmiquement espacées entre 0.5 et 70 Hz, puis on en déduit les échelles.

```python
f = np.logspace(np.log10(0.5), np.log10(70), 64)
echelles = pywt.frequency2scale("cmor1.5-1.0", f / fs)
```

L'espacement logarithmique est cohérent avec la nature multiplicative des bandes EEG : il donne autant de points à δ (0.5–4 Hz, un facteur 8) qu'à γ (30–70 Hz, un facteur 2.3).

#### Résolution adaptative

```
Haute fréquence (a petit)  : bonne résolution temporelle, mauvaise résolution fréquentielle
Basse fréquence (a grand)  : bonne résolution fréquentielle, mauvaise résolution temporelle
```

C'est exactement ce que l'on souhaite pour l'EEG : précision temporelle pour les transitoires gamma, précision fréquentielle pour les rythmes lents.

#### L'ondelette de Morlet complexe

L'ondelette de Morlet est le choix quasi-universel en neurosciences cognitives (Tallon-Baudry & Bertrand, 1999) : une sinusoïde modulée par une gaussienne, forme proche des bouffées oscillatoires neuronales.

```
ψ(t) = (1 / √(π·f_b)) · e^{j2π·f_c·t} · e^{−t² / f_b}
```

Le code utilise `cmor1.5-1.0`, c'est-à-dire **`f_b = 1.5`** (paramètre de largeur de bande) et **`f_c = 1.0` Hz** (fréquence centrale normalisée). Sa transformée de Fourier est une gaussienne centrée sur `f_c` :

```
Ψ(f) = e^{−π² · f_b · (f − f_c)²}
```

**Admissibilité.** Une ondelette mère doit être de moyenne nulle (`Ψ(0) = 0`). La Morlet complexe ne le vérifie qu'approximativement, la qualité de l'approximation dépendant du rapport `f_c / σ_f`. Pour `Ψ` ci-dessus, `σ_f = 1/(π√(2 f_b))`, donc avec `f_b = 1.5` :

```
σ_f ≈ 0.184    →    f_c / σ_f ≈ 5.4  > 5    ✓
```

La configuration retenue satisfait donc le critère usuel.

#### Scalogramme

```
SC(a, b) = |CWT{x(t)}(a, b)|²
```

C'est la représentation temps-échelle de la densité d'énergie, analogue au spectrogramme.

---

### 3.4 Comparaison STFT vs CWT

| Critère | STFT | CWT (Morlet) |
|---------|------|-------------|
| Résolution temporelle | Uniforme : `Δt = H/fs = 0.5 s` | Adaptative : meilleure à haute `f` |
| Résolution fréquentielle | Uniforme : `Δf = fs/N_w = 1 Hz` | Adaptative : meilleure à basse `f` |
| Hypothèse | Quasi-stationnarité par fenêtre | Aucune stationnarité requise |
| Complexité | O(N log N) par fenêtre | O(N · N_échelles) |
| Adapté aux transitoires | Moyen | Excellent |
| Usage dans ce projet | 5 énergies de bande + entropie + ratio γ/α | 5 énergies de bande |

Les deux représentations sont calculées pour **chaque** segment et leurs énergies de bande concaténées : elles sont complémentaires, pas concurrentes.

---

## 4. Génération des données synthétiques

`GenerateurDonneesEEG` produit des segments de **4 s à 256 Hz**, soit `N = 1024` points, à partir d'un `np.random.default_rng(42)` (résultats reproductibles).

### 4.1 Segment interictal

Somme de quatre oscillations à **phase aléatoire uniforme**, plus un bruit gaussien :

| Composante | Amplitude | Fréquence |
|-----------|-----------|-----------|
| δ | 0.8 | 2 Hz |
| θ | 0.5 | 6 Hz |
| α | 1.2 | 10 Hz |
| β | 0.3 | 20 Hz |
| bruit blanc | σ = 0.4 | — |

L'alpha domine, conformément à un EEG d'éveil calme.

### 4.2 Segment ictal

```
x_ictal(t) = [0.3·x_interictal(t) + γ_init(t) + γ_soutenu(t) + pointe_onde(t)] · e(t)
```

| Composante | Expression | Rôle physiologique |
|-----------|-----------|--------------------|
| Fond atténué | `0.3 · x_interictal(t)` | Effondrement des rythmes de base |
| Décharge gamma initiale | `2.5·sin(2π·40t)·e^{−0.3t}` | Bouffée rapide au début de crise |
| Gamma soutenu | `1.8·sin(2π·55t + 0.5)·(1 − e^{−0.8t})` | Hypersynchronisation installée |
| Pointe-onde | `3.0·sin(2π·3t)·\|sin(2π·3t)\|` | Complexe Spike-and-Wave à 3 Hz |
| Enveloppe | `linspace(0.5, 2.0)` | Croissance d'amplitude au cours du segment |

Le terme `sin(x)·|sin(x)|` produit une forme d'onde asymétrique et non sinusoïdale, plus riche en harmoniques qu'un simple sinus — c'est ce qui lui donne son allure « pointue ».

### 4.3 Jeu de données

`construire_jeu_de_donnees(nb_normaux=150, nb_crise=60)` → matrice `X` de forme **(210, 1024)** et vecteur `y` avec **28.6 % de positifs**.

> Les 210 segments sont **indépendants** : il n'y a ni signal continu, ni découpage avec recouvrement, ni notion de patient. Toute méthode de validation qui suppose une structure par patient (LOPO) est donc inapplicable en l'état.

---

## 5. Prétraitement du signal

### 5.1 Chaîne de traitement

```
x_brut(t) → [Butterworth passe-bande 0.5–70 Hz, ordre 4] → [Notch 50 Hz, Q = 30] → [z-score] → x_propre(t)
```

### 5.2 Filtre passe-bande Butterworth d'ordre 4

Le filtre de Butterworth est **maximalement plat** dans la bande passante (aucune ondulation, contrairement à Chebyshev ou elliptique) :

```
|H(f)|² = 1 / (1 + (f/fc)^{2N})
```

Pour `N = 4`, l'atténuation hors-bande est de **−80 dB/décade** (24 dB/octave). La bande [0.5, 70] Hz est choisie pour :

- **Borne inférieure 0.5 Hz :** éliminer la dérive de ligne de base (mouvements, respiration, transpiration) tout en conservant les ondes δ lentes ;
- **Borne supérieure 70 Hz :** conserver l'intégralité de la bande γ tout en rejetant les artéfacts musculaires haute fréquence.

**Implémentation réelle — forme SOS, pas (b, a).** Le code n'utilise pas `filtfilt(b, a, ...)` mais une cascade de sections du second ordre :

```python
self.sos_bp = scipy.signal.butter(ordre, [f_basse/nyq, f_haute/nyq],
                                  btype="band", output="sos")

def filtrer_signal(self, x):
    x = scipy.signal.sosfiltfilt(self.sos_bp, x)
    return scipy.signal.sosfiltfilt(self.sos_notch, x)
```

Le format SOS est numériquement bien plus stable que la forme directe `(b, a)` dès l'ordre 4, en particulier pour un passe-bande étroit en fréquence normalisée. Les fréquences sont passées **normalisées par Nyquist** (`f/nyq`), ce qui est cohérent avec l'absence d'argument `fs`.

**Filtrage zéro-phase.** `sosfiltfilt` applique le filtre en avant (délai de groupe `τ(f)`) puis en arrière (`−τ(f)`) : le délai net est nul à toutes les fréquences, ce qui est critique en EEG car un décalage de phase déplacerait les événements dans le temps. Le prix à payer est un doublement de l'ordre effectif (8 pôles), soit **−160 dB/décade**.

### 5.3 Filtre Notch à 50 Hz

Le réseau électrique européen rayonne à 50 Hz. Un coupe-bande IIR est appliqué, puis converti en SOS pour rester homogène avec le passe-bande :

```python
b, a = scipy.signal.iirnotch(f_reseau / nyq, Q=30)
self.sos_notch = scipy.signal.tf2sos(b, a)
```

Le facteur de qualité `Q = f0/Δf = 30` donne une largeur à −3 dB de `Δf ≈ 1.67 Hz`. La coupure reste donc largement à l'écart de la composante ictale à 55 Hz du générateur, qui n'est pas altérée.

### 5.4 Normalisation z-score

```
x_norm[n] = (x_propre[n] − μ_x) / (σ_x + ε)
```

Cette étape rend les descripteurs comparables entre sujets et sessions (les amplitudes EEG varient d'un facteur 10 selon l'épaisseur du crâne et la qualité du contact électrode-scalp).

**Conséquence importante pour le choix des descripteurs :** après z-score, `σ_x = 1` et `RMS(x) = 1` **par construction** pour tous les segments. Tout descripteur proportionnel à l'amplitude globale (RMS, variance, puissance totale, amplitude crête) devient une constante, de variance nulle, donc sans aucun pouvoir discriminant. C'est la raison pour laquelle le vecteur de descripteurs (section 6) ne contient **que des quantités invariantes d'échelle** : ratios d'énergie, kurtosis normalisé, rapport d'écarts-types intra-segment. La fonction `controler_descripteurs` sert précisément de garde-fou contre ce piège en signalant tout descripteur de variance quasi nulle.

---

## 6. Extraction des descripteurs

### 6.1 Énergies de bande

L'énergie de bande n'est pas estimée par la méthode de Welch mais **directement à partir des deux représentations temps-fréquence** déjà calculées. Pour chaque bande, on moyenne la puissance sur les bins de fréquence concernés **et** sur toutes les trames temporelles :

```
E_bande = mean_{(f, τ) : f ∈ bande}  P(f, τ)
```

```python
def energie_par_bande(self, f, P):
    ...
    masque = (f >= lo) & (f <= hi) if nom == derniere else (f >= lo) & (f < hi)
    energies[nom] = P[masque, :].mean()
```

Cette opération est appliquée deux fois : sur le spectrogramme STFT (`stft_*`) et sur le scalogramme CWT (`cwt_*`), soit **10 descripteurs**. Une bande vide (résolution fréquentielle insuffisante) lève une `ValueError` explicite plutôt que de produire silencieusement un `NaN`.

### 6.2 Ratio γ/α

```
R_γα = E_γ^STFT / (E_α^STFT + ε)        ε = 1e-12
```

C'est le marqueur le plus documenté de la littérature (Shoeb, 2009) : en interictal le spectre est dominé par α et le ratio reste faible ; en ictal l'énergie gamma explose par hypersynchronisation pendant que l'alpha s'effondre.

### 6.3 Entropie spectrale

Entropie de Shannon du spectre moyen normalisé en distribution de probabilité, `p(f) = S(f) / Σ S(f)` :

```
H_spec = −Σ_k p(f_k) · log₂(p(f_k))    [bits]
```

Le spectre est d'abord moyenné sur le temps (`P.mean(axis=1)`), puis normalisé.

**Interprétation :** un spectre plat (bruit blanc) a une entropie maximale `H_max = log₂(N_bins)`. Un signal ictal, bien que de forte amplitude, est **spectralement concentré** autour de ses fréquences de décharge → l'entropie **diminue** pendant la crise. Indicateur contre-intuitif mais robuste, et insensible à l'échelle du signal.

### 6.4 Descripteurs statistiques temporels

| Descripteur | Formule | Intérêt | Invariant d'échelle ? |
|-------------|---------|---------|----------------------|
| `aplatissement` (kurtosis) | `E[(x−μ)⁴] / (σ²)²` | Détecte les pointes (spikes) | ✅ oui |
| `variation_amplitude` | `std(x[N/2:]) / std(x[:N/2])` | Capte l'enveloppe croissante du segment ictal | ✅ oui |

`variation_amplitude` mesure directement l'effet de l'enveloppe `linspace(0.5, 2.0)` du générateur ictal. C'est un descripteur **spécifique aux données synthétiques** : sur de l'EEG réel, rien ne garantit qu'une crise soit croissante sur toute la fenêtre d'analyse (voir section 10).

> **Descripteurs volontairement absents :** le **RMS** est constant (= 1) après z-score, donc inutilisable ; le **taux de passage par zéro** (ZCR) n'est pas implémenté — il est largement redondant avec les énergies de bande gamma.

### 6.5 Vecteur complet — 14 descripteurs

L'ordre est celui de `ExtracteurCaracteristiques.NOMS` :

```
f = [stft_delta, stft_theta, stft_alpha, stft_beta, stft_gamma,     5 énergies STFT
     cwt_delta,  cwt_theta,  cwt_alpha,  cwt_beta,  cwt_gamma,      5 énergies CWT
     ratio_gamma_alpha,                                             1 ratio spectral
     entropie_spectrale,                                            1 entropie
     aplatissement,                                                 1 kurtosis
     variation_amplitude]                                           1 statistique d'enveloppe
```

La matrice finale a pour forme **(210, 14)**. `ExtracteurCaracteristiques.index_de(nom)` permet de retrouver la colonne d'un descripteur par son nom (utilisé pour les tests d'ablation en section 8).

---

## 7. Classification par SVM RBF

### 7.1 Formulation

Problème binaire. La formulation mathématique ci-dessous utilise la convention `y ∈ {−1, +1}` ; le code, lui, manipule les étiquettes scikit-learn `y ∈ {0, 1}` (0 = interictal, 1 = ictal).

Le vecteur `f ∈ ℝ¹⁴` est normalisé par `StandardScaler` **à l'intérieur d'un `Pipeline`** — point méthodologique essentiel : la moyenne et l'écart-type sont estimés uniquement sur le pli d'entraînement, ce qui évite toute fuite d'information vers le pli de test.

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True,
                class_weight="balanced", random_state=seed)),
])
```

`class_weight="balanced"` compense le déséquilibre 150/60 en pondérant chaque classe par l'inverse de son effectif.

### 7.2 Le noyau RBF

```
K(x, x') = exp(−γ · ‖x − x'‖²)    γ > 0
```

Par le théorème de Mercer, ce noyau est défini positif et correspond à un produit scalaire dans un espace de Hilbert de dimension infinie : `K(x, x') = ⟨φ(x), φ(x')⟩_H`. Le développement en série de l'exponentielle montre que le mapping implicite `φ` contient des monômes de **tous les degrés** — d'où la capacité à séparer des classes non linéairement séparables dans `ℝ¹⁴`.

`γ` contrôle la portée du noyau : grand `γ` → frontière très irrégulière (surapprentissage) ; petit `γ` → frontière lisse (sous-apprentissage).

### 7.3 Problème d'optimisation

```
min_{w, b, ξ}  ½‖w‖² + C · Σᵢ ξᵢ
sous :  yᵢ (⟨w, φ(xᵢ)⟩ + b) ≥ 1 − ξᵢ ,  ξᵢ ≥ 0
```

Formulation duale effectivement résolue :

```
max_{α}  Σᵢ αᵢ − ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ)
sous :   0 ≤ αᵢ ≤ C ,   Σᵢ αᵢ yᵢ = 0
```

Décision : `ŷ = sgn(Σᵢ αᵢ yᵢ K(xᵢ, x_test) + b)`. Seuls les **vecteurs supports** (`αᵢ > 0`) contribuent.

### 7.4 Grille d'hyperparamètres

Grille réellement explorée (`GRILLE_HYPERPARAMETRES`) :

```python
{
    "svm__C":     [0.1, 1, 10, 100],          # compromis marge / erreurs
    "svm__gamma": ["scale", "auto", 0.01, 0.1],  # portée du noyau RBF
}
```

`"scale"` vaut `1 / (n_features · Var(X))` et `"auto"` vaut `1 / n_features` : ce sont des heuristiques dépendantes des données, préférables à des valeurs fixes quand l'échelle des descripteurs peut changer.

La métrique optimisée est le **F1-score**, pas l'accuracy : avec 28.6 % de positifs ici — et souvent moins de 5 % en EEG clinique continu — l'accuracy récompenserait un modèle qui ne prédit jamais de crise.

---

## 8. Validation et résultats

### 8.1 Validation croisée imbriquée

Le protocole d'évaluation est une **validation croisée imbriquée** (`validation_imbriquee`) :

```
Boucle externe : StratifiedKFold(5)  →  estimation non biaisée de la performance
  └── Boucle interne : GridSearchCV(StratifiedKFold(5), scoring="f1")  →  choix de (C, γ)
```

Sélectionner les hyperparamètres puis rapporter le meilleur score de cette même sélection est un biais optimiste classique. La boucle externe évalue **la procédure complète** (recherche comprise), sur des segments jamais vus par la recherche.

Les deux boucles sont stratifiées, ce qui garantit la même proportion de crises dans chaque pli.

### 8.2 Références de comparaison

Trois garde-fous accompagnent le score principal :

1. **Classifieur naïf « classe majoritaire »** : ne prédit jamais de crise → F1 = 0 par construction. Utile pour rappeler que l'accuracy vaudrait déjà 71 % avec ce modèle.
2. **Classifieur naïf « tirage stratifié »** : prédit au hasard selon la distribution des classes → F1 ≈ 0.29, le vrai niveau du hasard.
3. **Ablations mono-descripteur** : le SVM est réentraîné sur **une seule colonne** de la matrice de descripteurs. Si un descripteur seul suffit à atteindre le score du modèle complet, le problème est trop facile.

### 8.3 Rapport hors-échantillon

Le rapport de classification final est produit par `cross_val_predict` : chaque prédiction provient d'un modèle qui n'a jamais vu le segment concerné. Un `fit(F, y)` suivi d'un `predict(F)` afficherait des scores d'entraînement, systématiquement optimistes et sans valeur d'évaluation.

### 8.4 Résultats obtenus

Sortie de `python code.py` (graine 42) :

```
Matrice de features : (210, 14)   (segments x descripteurs)
Classes : [150  60]  -> 29 % de crises
  [OK] tous les descripteurs ont une variance non nulle

F1 (CV imbriquee)          : 1.0000 +/- 0.0000
F1 (naif, classe majoritaire) : 0.0000
F1 (naif, tirage stratifie  ) : 0.2857

F1 (variation_amplitude  SEUL) : 1.0000
F1 (aplatissement        SEUL) : 0.9760
F1 (ratio_gamma_alpha    SEUL) : 1.0000
F1 (cwt_gamma            SEUL) : 1.0000
```

**Lecture honnête de ces chiffres.** Un F1 de 1.00 n'est pas une réussite : c'est le symptôme d'un problème **séparable par construction**. Les deux classes sont générées par deux formules déterministes différentes, dont l'une contient une composante à 40/55 Hz absente de l'autre. Les ablations le confirment sans ambiguïté — `ratio_gamma_alpha` **seul**, ou `cwt_gamma` **seul**, atteint déjà 1.00. Le SVM, le noyau RBF et les 14 descripteurs sont surdimensionnés pour cette tâche ; ils sont là pour valider la mécanique du pipeline, pas pour prouver une performance.

La seule conclusion défendable est : *le pipeline s'exécute de bout en bout, sans fuite de données, et retrouve la structure qu'on y a délibérément injectée.*

### 8.5 Métriques cliniques — non mesurées ici

En contexte clinique, les conséquences d'une **fausse alarme** et d'une **crise manquée** sont asymétriques :

```
Sensibilité (rappel) = TP / (TP + FN)     minimiser les crises manquées
Spécificité          = TN / (TN + FP)     minimiser les fausses alarmes
F1-score             = 2·P·R / (P + R)    compromis global
```

| Métrique | Seuil clinique usuel | Mesurable avec ce code ? |
|----------|---------------------|--------------------------|
| Sensibilité | > 90 % | ✅ (via `classification_report`) |
| Spécificité | > 95 % | ✅ |
| AUC-ROC | — | ⚠️ non calculée (le SVM expose pourtant `probability=True`) |
| Latence de détection | < 10 s | ❌ nécessite un signal continu horodaté |
| Taux de fausses alarmes | < 1/h | ❌ nécessite des heures d'enregistrement |

Les trois dernières lignes supposent un enregistrement continu réel : elles sont hors de portée d'un jeu de 210 segments indépendants.

### 8.6 Validation Leave-One-Patient-Out — à implémenter

Sur données réelles, la fuite inter-patient est le principal piège : un modèle entraîné et testé sur des segments du même patient apprend la signature de ce patient, pas celle d'une crise. La validation LOPO (entraîner sur N−1 patients, tester sur le restant) est alors indispensable. **Elle n'est pas implémentée** : le jeu synthétique ne comporte aucune notion de patient. Sa mise en place suppose l'intégration de CHB-MIT (section 11) et l'usage de `GroupKFold` / `LeaveOneGroupOut`.

---

## 9. Installation et utilisation

### 9.1 Installation

```bash
git clone https://github.com/<votre-compte>/detection-par-ondelette-des-epilepsie.git
cd detection-par-ondelette-des-epilepsie
pip install -r requirements.txt
```

**Dépendances effectivement importées par `code.py` :**

```
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
PyWavelets>=1.4      # pywt.frequency2scale requiert la version >= 1.4
```

Dépendances **optionnelles**, nécessaires uniquement aux extensions décrites en sections 9.3 et 11 (aucune n'est importée par le code actuel) :

```
matplotlib>=3.7      # visualisation (non implémentée)
mne>=1.5             # lecture EDF/EDF+ du dataset CHB-MIT (non implémentée)
```

### 9.2 Exécution

```bash
python code.py
```

Le script exécute la chaîne complète : génération → prétraitement → extraction → contrôle des descripteurs → validation imbriquée → références naïves → ablations → rapport hors-échantillon. Durée typique : quelques dizaines de secondes (la CWT sur 210 segments × 64 échelles domine le temps de calcul).

### 9.3 Utilisation programmatique

```python
from code import (GenerateurDonneesEEG, PretraiteurEEG,
                  ExtracteurCaracteristiques, validation_imbriquee)

X_brut, y  = GenerateurDonneesEEG(graine_aleatoire=7).construire_jeu_de_donnees(
                 nb_normaux=200, nb_crise=80)
X_propre   = PretraiteurEEG(fs=256).pretraiter_tous_segments(X_brut)
F          = ExtracteurCaracteristiques(fs=256).extraire_tous(X_propre)

print(F.shape)                        # (280, 14)
print(validation_imbriquee(F, y))     # F1 par pli externe
```

Accès aux représentations temps-fréquence brutes, par exemple pour tracer un spectrogramme ou un scalogramme (le tracé lui-même n'est pas fourni) :

```python
from code import AnalyseurTempsFrequence

tf = AnalyseurTempsFrequence(fs=256)
f_stft, t_stft, P_stft = tf.calculer_stft(X_propre[0])   # spectrogramme
f_cwt,  P_cwt          = tf.calculer_cwt(X_propre[0])    # scalogramme
```

---

## 10. Structure du projet et limites

### 10.1 Fichiers

```
.
├── code.py      # pipeline complet (générateur, prétraitement, TF, features, SVM, validation)
└── README.md    # ce document
```

Le projet tient en un seul module. Découpage naturel si le code venait à grossir : `generation.py`, `pretraitement.py`, `temps_frequence.py`, `descripteurs.py`, `classification.py`.

> ⚠️ Le nom `code.py` **masque le module `code` de la bibliothèque standard** (console interactive). L'import fonctionne depuis le répertoire du projet, mais un renommage en `eeg_pipeline.py` éviterait tout conflit — pensez alors à adapter les imports de la section 9.3.

### 10.2 Correspondance code ↔ documentation

| Classe / fonction | Section |
|-------------------|---------|
| `GenerateurDonneesEEG` | 4 |
| `PretraiteurEEG` | 5 |
| `AnalyseurTempsFrequence` | 3.2, 3.3, 6.1 |
| `ExtracteurCaracteristiques` | 6 |
| `pipeline_svm`, `GRILLE_HYPERPARAMETRES` | 7 |
| `validation_imbriquee`, `modele_avec_recherche` | 8.1 |

### 10.3 Limites assumées

1. **Données synthétiques.** Ni artéfacts oculaires ou musculaires, ni dérive de ligne de base, ni électrode décollée, ni variabilité inter-patient. Le prétraitement (sections 5.2–5.3) traite donc des problèmes que ces données ne présentent pas.
2. **Tâche triviale.** Un seul descripteur atteint F1 = 1.00 (section 8.4). Aucune conclusion sur le pouvoir discriminant relatif des descripteurs n'est possible ici.
3. **Descripteur non transposable.** `variation_amplitude` exploite une enveloppe croissante propre au générateur.
4. **Canal unique.** Aucune information spatiale, alors que la propagation inter-électrodes est un marqueur majeur des crises focales.
5. **Segments indépendants.** Ni continuité temporelle, ni latence de détection, ni taux de fausses alarmes par heure.
6. **Pas de visualisation.** Les figures spectrogramme / scalogramme restent à écrire.

### 10.4 Feuille de route

- [ ] Lecture EDF+ du dataset CHB-MIT via MNE (section 11)
- [ ] Découpage en segments glissants de 4 s avec 50 % de recouvrement, sur signal continu
- [ ] Validation `LeaveOneGroupOut` avec le patient comme groupe
- [ ] Métriques temporelles : latence de détection, fausses alarmes/heure
- [ ] Courbe ROC et AUC (le SVM est déjà configuré avec `probability=True`)
- [ ] Tracés spectrogramme / scalogramme (matplotlib)
- [ ] Extension multi-canaux

---

## 11. Extension : dataset CHB-MIT

> Cette section décrit une **piste d'extension**. Aucune ligne du code actuel ne lit ces données.

Le **CHB-MIT Scalp EEG Database** (Shoeb, 2009) est une référence académique pour la détection de crises, librement disponible sur PhysioNet :

- **23 patients** pédiatriques (5–22 ans), majoritairement pharmacorésistants
- **~916 heures** d'EEG continu
- **198 crises** annotées par des neurologues
- Format **EDF+**, 23 canaux, 256 Hz — la fréquence d'échantillonnage coïncide avec celle du générateur, ce qui rend le prétraitement réutilisable tel quel
- https://physionet.org/content/chbmit/1.0.0/

**Téléchargement :**

```bash
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/

# ou via le client PhysioNet
pip install wfdb
python -c "import wfdb; wfdb.dl_database('chbmit', './data/chb-mit/')"
```

**Lecture avec MNE-Python :**

```python
import mne
raw = mne.io.read_raw_edf("data/chb-mit/chb01/chb01_03.edf", preload=True)
x  = raw.get_data(picks=["FP1-F7"])[0]    # canal FP1-F7, signal 1D
fs = int(raw.info["sfreq"])               # 256 Hz
```

Le signal ainsi obtenu s'injecte directement dans `PretraiteurEEG` puis `ExtracteurCaracteristiques`, à condition d'ajouter en amont un découpage en fenêtres de 1024 points et l'attribution des étiquettes à partir des fichiers d'annotation `*.seizures`.

---

## 12. Références bibliographiques

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
| [10] | Varma, S. & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7:91. https://doi.org/10.1186/1471-2105-7-91 |
| [11] | Niedermeyer, E. & da Silva, F.L. (2004). *Electroencephalography: Basic Principles, Clinical Applications, and Related Fields* (5th ed.). Lippincott Williams & Wilkins. |
| [12] | Tzallas, A.T., Tsipouras, M.G. & Fotiadis, D.I. (2009). Epileptic Seizure Detection in EEGs Using Time–Frequency Analysis. *IEEE Trans. Information Technology in Biomedicine*, 13(5), 703–710. https://doi.org/10.1109/TITB.2009.2017939 |

> La référence [10] remplace la référence à Welch (1967) du document précédent, qui documentait une méthode d'estimation spectrale **non utilisée** par le code ; elle justifie en revanche directement le protocole de validation imbriquée de la section 8.1.