# SEARCH INDEX  (in Neovim: /@PCA_MAP then Enter, n = next.  List ALL tags: :g/#@ )
#   @PAGE_SETUP           page title and layout
#   @LOAD_MODELS          load models/bundle.joblib
#   @LOAD_SONGS           load models/songs.csv.gz
#   @FILTER_HELPER        turn a filter choice into a list
#   @FILTER_SONGS         include / exclude / search / sort / limit
#   @YOUTUBE_LINK         build the YouTube Music search link
#   @TABS                 the three tabs
#   @TAB_PREDICT          tab 1: predict a song
#   @SLIDERS              audio feature sliders and genre box
#   @PREDICT_PROBABILITY  hit probability from the best model
#   @MOOD_PREDICT         K-Means cluster + PCA point for the song
#   @SHOW_PREDICTION      show probability, mood, HIT / not hit
#   @PCA_MAP              draw the PCA mood map
#   @TAB_RESULTS          tab 2: model comparison table
#   @TAB_DISCOVER         tab 3: discover songs
#   @FILTERS_INCLUDE      mood / genre / language / country boxes
#   @FILTERS_EXCLUDE      hide moods / genres / languages / countries
#   @SEARCH_SORT_LIMIT    search box, sort order, how many
#   @RUN_FILTER           apply all filters
#   @RESULT_TABLE         table with Spotify / YouTube Music links

import joblib, numpy as np, pandas as pd
from urllib.parse import quote_plus
import matplotlib.pyplot as plt
import streamlit as st

#@PAGE_SETUP  page title and layout
st.set_page_config(page_title="Spotify Hit & Mood Engine", page_icon="🎧", layout="wide")

#@LOAD_MODELS  load models/bundle.joblib
@st.cache_resource
def load():
    return joblib.load("models/bundle.joblib")

b = load()
F = b["features"]

st.title("🎧 Spotify Hit & Mood Engine")
st.caption(f"Hit = popularity ≥ {b['hit_threshold']}. Classifier: {b['clf_name']} "
           f"(decision threshold {b['threshold']:.2f}).")

#@LOAD_SONGS  load models/songs.csv.gz
@st.cache_data
def load_songs():
    d = pd.read_csv("models/songs.csv.gz")
    if "country" not in d.columns:          # file made by an older train.py: retrain to get countries
        d["country"] = "Unknown"
    return d


#@FILTER_HELPER  turn a filter choice into a list
def _as_list(x):
    """Turn None / '' / 'All' / 'Any' / a single string / a list into a clean list ([] = no filter)."""
    if x is None or x in ("", "All", "Any"):
        return []
    return [x] if isinstance(x, str) else list(x)


#@FILTER_SONGS  include / exclude / search / sort / limit
def top_songs(songs, moods=None, genres=None, languages=None, countries=None,
              ex_moods=None, ex_genres=None, ex_languages=None, ex_countries=None,
              query="", sort="Highest hit score", n=25):
    """Keep songs that match every INCLUDE filter (empty = anything) and none of the EXCLUDE filters."""
    d = songs
    for col, inc, exc in [("mood", moods, ex_moods), ("track_genre", genres, ex_genres),
                          ("language", languages, ex_languages), ("country", countries, ex_countries)]:
        inc, exc = _as_list(inc), _as_list(exc)
        if inc:
            d = d[d[col].isin(inc)]
        if exc:
            d = d[~d[col].isin(exc)]
    if query.strip():
        q = query.strip().lower()
        d = d[d["track_name"].fillna("").str.lower().str.contains(q, regex=False)
              | d["artists"].fillna("").str.lower().str.contains(q, regex=False)]
    key = "hit_prob" if sort == "Highest hit score" else "popularity"
    ordered = d.sort_values(key, ascending=False)
    return ordered if n is None else ordered.head(n)   # n=None -> every matching song


#@YOUTUBE_LINK  build the YouTube Music search link
def yt_music_link(song, artists):
    """YouTube Music has no track ids in our data, so link to a search for 'song + first artist'."""
    first_artist = str(artists).split(";")[0]
    return "https://music.youtube.com/search?q=" + quote_plus(f"{song} {first_artist}")


#@TABS  the three tabs
tab1, tab2, tab3 = st.tabs(["Predict a song", "Model results", "Discover songs"])

#@TAB_PREDICT  tab 1: predict a song
with tab1:
    left, right = st.columns([1, 1.2])
    vals = {}
    with left:
        #@SLIDERS  audio feature sliders and genre box
        st.subheader("Audio features")
        genre = st.selectbox("Genre", b["genres"])
        for f in b["ui_features"]:
            lo, hi, med = b["ranges"][f]
            if f == "explicit":
                vals[f] = int(st.checkbox("Explicit", value=False))
            elif f == "mode":
                vals[f] = 1 if st.radio("Mode", ["Major", "Minor"], horizontal=True) == "Major" else 0
            elif f in ("key", "time_signature"):
                vals[f] = st.slider(f.replace("_", " ").capitalize(), int(lo), int(hi), int(med))
            elif hi <= 1.0 and lo >= 0:
                vals[f] = st.slider(f.capitalize(), 0.0, 1.0, float(np.clip(med, 0, 1)), 0.01)
            else:
                vals[f] = st.slider(f.replace("_", " ").capitalize(), float(lo), float(hi), float(med))
        vals["genre_score"] = b["genre_map"].get(genre, b["genre_prior"])

    #@PREDICT_PROBABILITY  hit probability from the best model
    x = pd.DataFrame([vals])[F]
    xs = b["scaler"].transform(x)
    prob = float(b["clf"].predict_proba(xs)[0, 1])
    #@MOOD_PREDICT  K-Means cluster + PCA point for the song
    xm = xs[:, b["mood_idx"]]
    cl = int(b["kmeans"].predict(xm)[0])
    pt = b["pca"].transform(xm)[0]

    with right:
        #@SHOW_PREDICTION  show probability, mood, HIT / not hit
        st.subheader("Prediction")
        c1, c2 = st.columns(2)
        c1.metric("Hit probability", f"{prob:.0%}")
        c2.markdown("**Mood cluster**")
        c2.markdown(f"### {b['cluster_names'][cl]}")
        st.progress(min(prob, 1.0))
        if prob >= b["threshold"]:
            st.success("Likely a HIT 🔥")
        else:
            st.info("Probably not a hit")

        #@PCA_MAP  draw the PCA mood map
        mp = b["map_points"]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for c in sorted(mp.cluster.unique()):
            m = mp.cluster == c
            ax.scatter(mp.pc1[m], mp.pc2[m], s=4, alpha=.35, label=b["cluster_names"][c])
        ax.scatter(*pt, marker="*", s=300, c="black", edgecolors="white", label="Your song", zorder=5)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(markerscale=3, fontsize=7)
        st.pyplot(fig)

#@TAB_RESULTS  tab 2: model comparison table
with tab2:
    st.subheader("Model comparison (test set)")
    st.dataframe(b["results"], hide_index=True)
    st.caption("F1 matters more than accuracy here because hits are the minority class. "
               "Each model uses its own decision threshold, chosen on training data only.")

#@TAB_DISCOVER  tab 3: discover songs
with tab3:
    st.subheader("Discover songs")
    songs = load_songs()
    moods_all = sorted(b["cluster_names"].values())
    genres_all = sorted(songs["track_genre"].dropna().unique())
    langs_all = sorted(songs["language"].dropna().unique())
    countries_all = sorted(songs["country"].dropna().unique())
    countries_known = [c for c in countries_all if c != "Unknown"]
    langs_known = [l for l in langs_all if not l.startswith("Other")]

    #@FILTERS_INCLUDE  mood / genre / language / country boxes
    st.markdown("**Show songs that match** (leave a box empty to allow anything)")
    f1, f2, f3, f4 = st.columns(4)
    mood_pick = f1.multiselect("Mood", moods_all, placeholder="Any mood")
    genre_pick = f2.multiselect("Genre", genres_all, placeholder="Any genre")
    lang_pick = f3.multiselect("Language", langs_known, placeholder="Any language")
    country_pick = f4.multiselect("Country (approx.)", countries_known, placeholder="Any country")

    #@FILTERS_EXCLUDE  hide moods / genres / languages / countries
    with st.expander("Hide songs from these categories (exclude)"):
        x1, x2, x3, x4 = st.columns(4)
        ex_mood = x1.multiselect("Hide moods", moods_all, key="ex_mood")
        ex_genre = x2.multiselect("Hide genres", genres_all, key="ex_genre")
        ex_lang = x3.multiselect("Hide languages", langs_all, key="ex_lang")
        ex_country = x4.multiselect("Hide countries", countries_all, key="ex_country")

    #@SEARCH_SORT_LIMIT  search box, sort order, how many
    g1, g2, g3 = st.columns([2, 1.3, 1])
    query = g1.text_input("Search song or artist (optional)")
    sort_by = g2.radio("Sort by", ["Highest hit score", "Most popular"], horizontal=True)
    n = g3.number_input("How many", 5, 1000, 25, step=5)
    show_all = g3.checkbox("Show all matches")

    #@RUN_FILTER  apply all filters
    matches = top_songs(songs, mood_pick, genre_pick, lang_pick, country_pick,
                        ex_mood, ex_genre, ex_lang, ex_country, query, sort_by, None)
    res = matches if show_all else matches.head(int(n))
    if res.empty:
        st.warning("No songs match these filters. Try loosening one of them.")
    else:
        st.write(f"**{len(matches):,}** songs match your filters, showing **{len(res):,}**.")
        #@RESULT_TABLE  table with Spotify / YouTube Music links
        out = pd.DataFrame({
            "Song": res["track_name"], "Artist": res["artists"], "Genre": res["track_genre"],
            "Mood": res["mood"], "Language": res["language"], "Country": res["country"],
            "Popularity": res["popularity"], "Hit score": (res["hit_prob"] * 100).round(0),
            "Spotify": "https://open.spotify.com/track/" + res["track_id"],
            "YouTube Music": [yt_music_link(s, a) for s, a in zip(res["track_name"], res["artists"])],
        })
        st.dataframe(out, hide_index=True, use_container_width=True, column_config={
            "Spotify": st.column_config.LinkColumn("Spotify", display_text="Spotify ▶"),
            "YouTube Music": st.column_config.LinkColumn("YouTube Music", display_text="YT Music ▶"),
            "Hit score": st.column_config.NumberColumn("Hit score (%)"),
        })
    st.caption("Spotify opens the exact track; YouTube Music opens a search for the song and artist. "
               "Hit score = the model's predicted chance of being a hit. The dataset has no language or "
               "country column, so both are inferred: language from tags like '(Tamil)' in the album or song "
               "name (else the genre), country from an unambiguous language tag or a country-specific genre "
               "such as k-pop. Songs we cannot place are 'Unknown'/'Other', so country filters cover only part of the data.")
