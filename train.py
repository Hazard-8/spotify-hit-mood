"""
Spotify Hit & Mood Engine - training pipeline
=============================================
HOW TO RUN:   python train.py            (needs dataset.csv in this folder)
WHAT IT MAKES: models/bundle.joblib  (all trained models, used by app.py)
               models/songs.csv.gz   (song list for the "Discover songs" tab)
               models/results.csv + two PNG plots (use these in your report)

THE TWO PARTS OF THE PROJECT
  SUPERVISED   (learns from labels)  : Logistic Regression, KNN, Tuned Gradient Boosting
                                       -> predict "hit" or "not a hit"
  UNSUPERVISED (no labels needed)    : PCA (2D map) and K-Means (mood clusters)

SEARCH INDEX  (in Neovim: /@KNN then Enter, n = next.  List ALL tags: :g/#@ )
  @SETTINGS             hit threshold, random seed, file name
  @FEATURE_LIST         which columns the models use
  @LANGUAGE_RULES       tags/words used to detect a song's language
  @COUNTRY_RULES        language/genre -> country tables
  @LOAD_DATA            read dataset.csv
  @DROPNA               remove rows with no id / popularity
  @FILLNA               fill missing text and numbers
  @DUPLICATES           one row per track
  @JUNK_ROWS            remove tempo 0 / very long songs
  @HIT_LABEL            hit = popularity >= threshold
  @TRAIN_TEST_SPLIT     80% train / 20% test
  @GENRE_SCORE          how often each genre produces hits
  @SCALER               StandardScaler (same range for all features)
  @SUPERVISED           SUPERVISED LEARNING section (hit / not hit)
  @THRESHOLD            choose the best hit cutoff per model
  @EVALUATE             accuracy, F1, ROC-AUC on the test set
  @LOGISTIC_REGRESSION  supervised model 1
  @KNN                  supervised model 2 (K-Nearest Neighbours)
  @GRADIENT_BOOSTING    supervised model 3
  @TUNING               RandomizedSearchCV hyper-parameter search
  @BEST_MODEL           compare models, keep the best by F1
  @UNSUPERVISED         UNSUPERVISED LEARNING section (moods)
  @PCA                  reduce 9 audio features to 2D
  @KMEANS               group songs into mood clusters
  @SILHOUETTE           choose k with the silhouette score
  @CLUSTER_NAMES        name clusters (Happy & Energetic ...)
  @SONG_TABLE           song list for the Discover tab
  @HIT_SCORE            predicted hit chance for every song
  @LANGUAGE             detect each song's language
  @COUNTRY              infer each song's country
  @SONG_FILE            write models/songs.csv.gz
  @SAVE_OUTPUTS         plots + model bundle
  @PCA_PLOT             PCA map picture
  @SILHOUETTE_PLOT      picture of how k was chosen
  @SAVE_BUNDLE          write models/bundle.joblib
"""
import re
import sys
from collections import Counter
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")                      # draw plots to files, no window needed
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, silhouette_score, roc_auc_score

#@SETTINGS  hit threshold, random seed, file name
# ======================================================================
# SETTINGS
# ======================================================================
CSV = sys.argv[1] if len(sys.argv) > 1 else "dataset.csv"
HIT_THRESHOLD = 65   # a song with popularity >= 65 is called a "hit" (explain this choice in your report)
SEED = 42            # fixed random seed so results are the same every run

#@FEATURE_LIST  which columns the models use
# The 9 audio columns that describe how a song sounds. Used for the mood clusters.
AUDIO = ["danceability", "energy", "loudness", "speechiness", "acousticness",
         "instrumentalness", "liveness", "valence", "tempo"]

# Features given to the hit classifier = audio + a few extras.
# "genre_score" is created in STEP 2 (how often songs of that genre become hits).
FEATURES = AUDIO + ["duration_min", "explicit", "mode", "key", "time_signature", "genre_score"]

#@LANGUAGE_RULES  tags/words used to detect a song's language
# --- Language detection (used only for the "Discover songs" tab) ---
# The dataset has no language column, but many album names carry a tag like
# "Brahmastra (Malayalam) [Original Motion Picture Soundtrack]". We read that tag.
# Indian languages are matched as a whole WORD anywhere in the album or song name
# (e.g. "Malayalam Hits", "Kuthu (Tamil Version)", "From 'X (Malayalam)'").
INDIAN_LANGS = ["Malayalam", "Tamil", "Telugu", "Kannada", "Hindi", "Bengali", "Marathi",
                "Punjabi", "Gujarati", "Bhojpuri", "Urdu"]
WORD_REGEX = re.compile(r"\b(" + "|".join(INDIAN_LANGS) + r")\b", re.IGNORECASE)
# Other languages are only trusted when written as a tag in brackets, e.g. "(Korean)" or "[French Version]".
# (A plain word like "French" could just be part of a title such as "French Kiss".)
OTHER_LANGS = ["Korean", "Japanese", "Spanish", "French", "German", "Portuguese", "Arabic",
               "Turkish", "Mandarin", "Chinese", "Cantonese", "Thai", "Indonesian", "Malay",
               "Italian", "Russian"]
TAG_REGEX = re.compile(r"[\(\[]\s*(" + "|".join(OTHER_LANGS) + r")(?:\s+version)?\s*[\)\]]",
                       re.IGNORECASE)

# Fallback when there is no tag: guess the language from the genre name.
LANG_MAP = {
    "k-pop": "Korean", "j-pop": "Japanese", "j-rock": "Japanese", "j-dance": "Japanese",
    "j-idol": "Japanese", "anime": "Japanese", "mandopop": "Mandarin", "cantopop": "Cantonese",
    "indian": "Indian (not tagged)", "malay": "Malay", "iranian": "Persian",
    "french": "French", "german": "German", "swedish": "Swedish", "turkish": "Turkish",
    "spanish": "Spanish", "latin": "Spanish", "latino": "Spanish", "salsa": "Spanish",
    "tango": "Spanish", "reggaeton": "Spanish",
    "brazil": "Portuguese", "mpb": "Portuguese", "pagode": "Portuguese", "samba": "Portuguese",
    "sertanejo": "Portuguese", "forro": "Portuguese",
}

#@COUNTRY_RULES  language/genre -> country tables
# --- Country (approximate) ---
# The dataset has NO country column either. We infer a country only where it is fairly safe:
#   1) an unambiguous language tag (e.g. "(Malayalam)" -> India), else
#   2) a genre that is tied to one country (e.g. "k-pop" -> South Korea).
# Everything else (pop, rock, English-language genres...) stays "Unknown".
LANG_COUNTRY = {
    "Malayalam": "India", "Tamil": "India", "Telugu": "India", "Kannada": "India",
    "Hindi": "India", "Marathi": "India", "Gujarati": "India", "Bhojpuri": "India",
    "Korean": "South Korea", "Japanese": "Japan", "Turkish": "Turkey", "Thai": "Thailand",
    "Indonesian": "Indonesia", "Malay": "Malaysia", "Italian": "Italy", "Russian": "Russia",
    "Cantonese": "Hong Kong",
}
GENRE_COUNTRY = {
    "indian": "India", "british": "United Kingdom", "french": "France", "german": "Germany",
    "swedish": "Sweden", "turkish": "Turkey", "iranian": "Iran", "malay": "Malaysia",
    "k-pop": "South Korea", "j-pop": "Japan", "j-rock": "Japan", "j-dance": "Japan",
    "j-idol": "Japan", "anime": "Japan", "cantopop": "Hong Kong", "mandopop": "China / Taiwan",
    "brazil": "Brazil", "mpb": "Brazil", "pagode": "Brazil", "samba": "Brazil",
    "sertanejo": "Brazil", "forro": "Brazil", "tango": "Argentina",
}

# ======================================================================
# STEP 1: LOAD AND CLEAN THE DATA
# Real data is messy. We fix problems before any model sees it.
# ======================================================================
#@LOAD_DATA  read dataset.csv
df = pd.read_csv(CSV)
print("Raw shape:", df.shape)                                   # (rows, columns)
print("\nNull counts BEFORE cleaning:\n", df.isnull().sum()[df.isnull().sum() > 0])

# Remove the useless row-number column that Kaggle's CSV adds ("Unnamed: 0")
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

#@DROPNA  remove rows with no id / popularity
# 1a) dropna: a row with no id or no popularity is unusable.
#     We can't guess popularity because it is what defines our label.
df = df.dropna(subset=["track_id", "popularity"])

#@FILLNA  fill missing text and numbers
# 1b) fillna: missing TEXT -> "Unknown"; missing NUMBERS -> the median of that column.
#     (Median is safer than mean because extreme values don't distort it.)
for col in ["artists", "album_name", "track_name", "track_genre"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")
num_cols = AUDIO + ["duration_ms", "mode", "key", "time_signature"]
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df["explicit"] = df["explicit"].fillna(False)

#@DUPLICATES  one row per track
# 1c) Remove duplicates. The same track appears once per genre it belongs to,
#     so we keep one row per track_id (the one with the highest popularity).
df = df.sort_values("popularity", ascending=False).drop_duplicates("track_id")

#@JUNK_ROWS  remove tempo 0 / very long songs
# 1d) Remove junk rows and make helper columns.
df["duration_min"] = df["duration_ms"] / 60000      # milliseconds -> minutes
df["explicit"] = df["explicit"].astype(int)         # True/False -> 1/0
df = df[(df["tempo"] > 0) & (df["duration_min"] < 15)]   # tempo 0 or 15+ min songs are bad data

print("\nNull counts AFTER cleaning:", int(df.isnull().sum().sum()))   # should be 0

#@HIT_LABEL  hit = popularity >= threshold
# 1e) Create the label we want to predict: 1 = hit, 0 = not a hit.
#     IMPORTANT: popularity is only used to make this label. It is NEVER a feature,
#     otherwise the model could just read the answer (that would be "label leakage").
df["hit"] = (df["popularity"] >= HIT_THRESHOLD).astype(int)
print(f"Clean shape: {df.shape} | hit rate: {df['hit'].mean():.1%}")    # hits are rare -> imbalanced

#@TRAIN_TEST_SPLIT  80% train / 20% test
# ======================================================================
# STEP 2: TRAIN/TEST SPLIT + GENRE SCORE
# We hide 20% of songs (the test set) to check how well the model works on songs it never saw.
# ======================================================================
tr_idx, te_idx = train_test_split(df.index, test_size=0.2, stratify=df["hit"], random_state=SEED)
# stratify keeps the same hit-rate in both train and test sets

#@GENRE_SCORE  how often each genre produces hits
# Genre score = "what fraction of songs in this genre are hits".
# It is computed from TRAIN songs only, so the test set stays truly unseen.
prior = df.loc[tr_idx, "hit"].mean()                               # overall hit rate
g = df.loc[tr_idx].groupby("track_genre")["hit"].agg(["sum", "count"])
M = 20                                                             # smoothing strength
# smoothing pulls genres with few songs toward the overall hit rate (avoids noisy scores)
genre_map = ((g["sum"] + M * prior) / (g["count"] + M)).to_dict()
df["genre_score"] = df["track_genre"].map(genre_map).fillna(prior)

X, y = df[FEATURES], df["hit"]
X_tr, X_te, y_tr, y_te = X.loc[tr_idx], X.loc[te_idx], y.loc[tr_idx], y.loc[te_idx]

#@SCALER  StandardScaler (same range for all features)
# Scaling: put every feature on a similar range (tempo is ~120, valence is 0-1).
# Needed for KNN and Logistic Regression. The scaler learns ONLY from train data.
scaler = StandardScaler().fit(X_tr)
Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)

#@SUPERVISED  SUPERVISED LEARNING section (hit / not hit)
# ======================================================================
# STEP 3: SUPERVISED LEARNING - predict hit / not hit
# We train 3 models and compare them with accuracy, F1 and ROC-AUC.
#   accuracy = % of correct answers (misleading here: always saying "no hit" gives ~94%)
#   F1       = balance of precision and recall for the hit class (the number that matters)
#   ROC-AUC  = how well the model ranks hits above non-hits (1.0 perfect, 0.5 random)
# ======================================================================
results, thresholds = [], {}

#@THRESHOLD  choose the best hit cutoff per model
def pick_threshold(model):
    """Models output a probability. We call it a hit if probability >= threshold.
    Instead of always using 0.5, find the cutoff with the best F1.
    We use cross-validation on TRAIN data only, so the test set is never touched."""
    oof = cross_val_predict(clone(model), Xs_tr, y_tr, cv=3, method="predict_proba")[:, 1]
    grid = np.arange(0.05, 0.95, 0.01)
    return float(grid[int(np.argmax([f1_score(y_tr, oof >= t) for t in grid]))])

#@EVALUATE  accuracy, F1, ROC-AUC on the test set
def evaluate(name, model):
    """Score a trained model on the hidden test set and store the result."""
    thr = pick_threshold(model)
    proba = model.predict_proba(Xs_te)[:, 1]      # predicted chance of being a hit
    pred = proba >= thr                           # turn probability into hit / not hit
    thresholds[name] = thr
    results.append({"model": name, "threshold": round(thr, 2),
                    "accuracy": accuracy_score(y_te, pred),
                    "f1": f1_score(y_te, pred), "roc_auc": roc_auc_score(y_te, proba)})
    print(results[-1])

#@LOGISTIC_REGRESSION  supervised model 1
# 3a) Logistic Regression: simple baseline, draws a straight boundary between hits and non-hits.
#     class_weight="balanced" makes mistakes on the rare hit class count more.
logreg = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xs_tr, y_tr)
evaluate("Logistic Regression", logreg)

#@KNN  supervised model 2 (K-Nearest Neighbours)
# 3b) KNN: to judge a song, look at the 15 most similar training songs and let them vote.
knn = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1).fit(Xs_tr, y_tr)
evaluate("KNN (k=15)", knn)

#@GRADIENT_BOOSTING  supervised model 3
# 3c) Gradient Boosting: builds many small decision trees, each fixing the previous ones' mistakes.
#     "Tuned" = RandomizedSearchCV tries 15 random setting combinations and keeps the best by F1.
#@TUNING  RandomizedSearchCV hyper-parameter search
search = RandomizedSearchCV(
    HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
    {"learning_rate": [0.03, 0.05, 0.1, 0.2], "max_depth": [3, 5, 7, None],
     "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0, 0.1, 1, 10],
     "min_samples_leaf": [20, 50, 100], "max_iter": [200, 300, 500]},
    n_iter=15, scoring="f1", cv=3, random_state=SEED, n_jobs=-1)
search.fit(Xs_tr, y_tr)
print("Best boosting params:", search.best_params_)
evaluate("Tuned Boosting", search.best_estimator_)

#@BEST_MODEL  compare models, keep the best by F1
# 3d) Compare the three models and keep the winner (highest F1) for the app.
res = pd.DataFrame(results).round(4)
res.to_csv("models/results.csv", index=False)
print("\n", res.to_string(index=False))
best_name = res.sort_values("f1", ascending=False).iloc[0]["model"]
clf = {"Logistic Regression": logreg, "KNN (k=15)": knn, "Tuned Boosting": search.best_estimator_}[best_name]
print("Best model by F1:", best_name, "| threshold:", round(thresholds[best_name], 2))

#@UNSUPERVISED  UNSUPERVISED LEARNING section (moods)
# ======================================================================
# STEP 4: UNSUPERVISED LEARNING - mood clusters and 2D map
# No labels here. The algorithms look only at the audio features and find structure.
# ======================================================================
Xs_all = scaler.transform(X)                      # scaled features for ALL songs
mood_idx = [FEATURES.index(c) for c in AUDIO]     # positions of the 9 audio columns
Xmood = Xs_all[:, mood_idx]                       # only the audio features describe "mood"

#@PCA  reduce 9 audio features to 2D
# 4a) PCA: squeeze 9 audio dimensions into 2 so we can draw a map of songs.
pca = PCA(n_components=2, random_state=SEED).fit(Xs_tr[:, mood_idx])
print("PCA explained variance:", pca.explained_variance_ratio_.round(3))   # how much info 2D keeps

#@KMEANS  group songs into mood clusters
# 4b) K-Means: groups songs into k clusters of similar sound.
#     Which k? Try 2..8 and use the silhouette score (higher = cleaner clusters).
#     Silhouette on all songs is slow, so use a random sample of 8000.
#@SILHOUETTE  choose k with the silhouette score
rng = np.random.RandomState(SEED)
samp = rng.choice(len(Xmood), size=min(8000, len(Xmood)), replace=False)
sil = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit(Xmood)
    sil[k] = silhouette_score(Xmood[samp], km.labels_[samp])
best_k = max(sil, key=sil.get)
K = best_k if best_k >= 4 else 4          # silhouette often picks k=2; we use at least 4 for richer moods
print("Silhouette by k:", {k: round(v, 3) for k, v in sil.items()}, "-> using k =", K)
kmeans = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Xmood)
df["cluster"] = kmeans.labels_            # every song now has a cluster number

#@CLUSTER_NAMES  name clusters (Happy & Energetic ...)
# 4c) Give each cluster a human name from its average energy and valence (happiness).
prof = df.groupby("cluster")[["energy", "valence", "acousticness", "danceability"]].mean()
med = df[["energy", "valence"]].median()
names = {}
for c, r in prof.iterrows():
    e = "Energetic" if r.energy >= med.energy else "Calm"
    v = "Happy" if r.valence >= med.valence else "Moody"
    names[c] = f"{v} & {e}"
# If two clusters got the same name, add the trait that makes each one most different.
traits = ["danceability", "acousticness", "speechiness", "liveness", "instrumentalness"]
tm = df.groupby("cluster")[traits].mean()
z = (tm - df[traits].mean()) / df[traits].std()
for base, cnt in Counter(names.values()).items():
    if cnt > 1:
        for c in [k for k, v in names.items() if v == base]:
            t = z.loc[c].abs().idxmax()
            names[c] = f"{base} ({'High' if z.loc[c, t] > 0 else 'Low'} {t})"
for base, cnt in Counter(names.values()).items():     # last resort: number them
    if cnt > 1:
        for i, c in enumerate([k for k, v in names.items() if v == base], 1):
            names[c] = f"{base} #{i}"
print(prof.round(2).assign(name=pd.Series(names)))

#@SONG_TABLE  song list for the Discover tab
# ======================================================================
# STEP 5: BUILD THE SONG LIST FOR THE "DISCOVER SONGS" TAB
# ======================================================================
#@HIT_SCORE  predicted hit chance for every song
# 5a) Hit score for every song = the best model's predicted chance of being a hit.
df["hit_prob"] = clf.predict_proba(Xs_all)[:, 1]

#@LANGUAGE  detect each song's language
# 5b) Language: first look for a tag like "(Tamil)" in the album or song name.
#     If there is none, fall back to guessing from the genre name; otherwise "Other".
text = df["album_name"] + " " + df["track_name"]
tag = text.str.extract(WORD_REGEX, expand=False).fillna(text.str.extract(TAG_REGEX, expand=False)).str.title()

# The same song often exists twice: once on a tagged album ("... (Malayalam)") and once on an
# untagged single or compilation. When we later keep only one copy per song, we must not lose
# the tag, so every copy inherits the tag found on ANY copy of that (song, artist).
tagged = df[tag.notna()].assign(tag=tag[tag.notna()]).drop_duplicates(["track_name", "artists"])
pair_lang = {(s, a): l for s, a, l in zip(tagged["track_name"], tagged["artists"], tagged["tag"])}
inherited = pd.Series([pair_lang.get(k) for k in zip(df["track_name"], df["artists"])], index=df.index)
print("Songs that regained a language tag from another copy of the same song:",
      int((tag.isna() & inherited.notna()).sum()))

df["language"] = tag.fillna(inherited).fillna(df["track_genre"].map(LANG_MAP)).fillna("Other / not specified")

#@COUNTRY  infer each song's country
# 5b-2) Country: unambiguous language tag first, then country-specific genre, else "Unknown".
own_lang = tag.fillna(inherited)
df["country"] = own_lang.map(LANG_COUNTRY).fillna(df["track_genre"].map(GENRE_COUNTRY)).fillna("Unknown")
print("\nCountry counts (top 12):\n", df["country"].value_counts().head(12))
print("\nLanguage counts (top 15):\n", df["language"].value_counts().head(15))

#@SONG_FILE  write models/songs.csv.gz
# 5c) Keep the columns the app needs; one row per (song name, artist).
songs = df[["track_id", "track_name", "artists", "track_genre", "cluster",
            "popularity", "hit_prob", "language", "country"]].copy()
songs["mood"] = songs["cluster"].map(names)
songs = songs.sort_values("popularity", ascending=False).drop_duplicates(["track_name", "artists"])
songs["hit_prob"] = songs["hit_prob"].round(3)
songs.drop(columns="cluster").to_csv("models/songs.csv.gz", index=False, compression="gzip")
print("Saved models/songs.csv.gz:", len(songs), "songs")
print("\nIndian-language songs available in the app (after removing same-name duplicates):")
for lang in INDIAN_LANGS:
    print(f"  {lang:10s} {int((songs['language'] == lang).sum())}")

#@SAVE_OUTPUTS  plots + model bundle
# ======================================================================
# STEP 6: SAVE PLOTS AND THE MODEL BUNDLE
# ======================================================================
#@PCA_PLOT  PCA map picture
# 6a) Scatter plot of the PCA map, coloured by mood cluster (put this in your report).
coords = pca.transform(Xmood[samp])
plt.figure(figsize=(7, 5))
for c in range(K):
    m = df["cluster"].values[samp] == c
    plt.scatter(coords[m, 0], coords[m, 1], s=4, alpha=.4, label=names[c])
plt.legend(markerscale=4, fontsize=7); plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA map of tracks by mood cluster")
plt.tight_layout(); plt.savefig("models/pca_clusters.png", dpi=150); plt.close()

#@SILHOUETTE_PLOT  picture of how k was chosen
# 6b) Plot showing how we chose k for K-Means.
plt.figure(figsize=(5, 3)); plt.plot(list(sil), list(sil.values()), "o-")
plt.xlabel("k"); plt.ylabel("silhouette"); plt.title("Choosing k"); plt.tight_layout()
plt.savefig("models/silhouette.png", dpi=150); plt.close()

#@SAVE_BUNDLE  write models/bundle.joblib
# 6c) Save everything app.py needs in ONE file, so the app applies exactly the same
#     scaling, PCA, clustering and classifier that we used here.
UI_FEATURES = [f for f in FEATURES if f != "genre_score"]    # features the user sets with sliders
bundle = {
    "features": FEATURES, "ui_features": UI_FEATURES, "mood_idx": mood_idx, "scaler": scaler,
    "pca": pca, "kmeans": kmeans, "cluster_names": names, "clf": clf, "clf_name": best_name,
    "threshold": thresholds[best_name], "results": res, "hit_threshold": HIT_THRESHOLD,
    "genre_map": genre_map, "genre_prior": float(prior), "genres": sorted(genre_map),
    "languages": sorted(l for l in songs["language"].unique() if not l.startswith("Other")),
    "countries": sorted(c for c in songs["country"].unique() if c != "Unknown"),
    "map_points": pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1],
                                "cluster": df["cluster"].values[samp]}),
    "ranges": {c: (float(X[c].min()), float(X[c].max()), float(X[c].median())) for c in UI_FEATURES},
}
joblib.dump(bundle, "models/bundle.joblib", compress=3)
print("Saved models/bundle.joblib")
