import pandas as pd
import snowflake.connector
from sqlalchemy import create_engine
from sqlalchemy.dialects import *
import sqlalchemy
import re
import spacy
from spacy.lang.en import English
from spacy import displacy
from spacy.pipeline import EntityRuler
import probablepeople as pp
import string
import warnings
import textdistance
import dateutil
import datetime
import numpy as np
from scipy import stats
from datetime import date
from io import StringIO
import psycopg2
from collections import Counter
import io
import matplotlib.pyplot as plt
from autocorrect import Speller
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec
from gensim.test.utils import common_texts, get_tmpfile
import gensim.downloader as api
import gensim
from gensim.models import Phrases
import gensim.models.keyedvectors as sent2vec
import pyemd
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import re
from surprise import Reader, Dataset, SVD
from rake_nltk import Rake
from rake_nltk import Metric, Rake
import string
from difflib import SequenceMatcher
from nltk.stem.snowball import SnowballStemmer
from pytrends.request import TrendReq
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from pytrends.request import TrendReq
import time 
import os 
import glob
import warnings
warnings.simplefilter("ignore")

from scipy import stats

nltk.download("stopwords")
warnings.simplefilter("ignore")
# %load_ext nb_black





################# Read in Data ##################################
omitted 



############## Cleaning and processing ###########################################
# Cleaning up categories to remove words like Kindle and Books, since we they aren't relevant categories to us
def clean_cats(df):
    df.cat = df.cat.apply(lambda x: x.strip("\n[]()-"))
    df.cat = df.cat.apply(lambda x: x.replace("\n", ""))
    df.cat = df.cat.apply(lambda x: x.replace('"', ""))
    df.cat = df.cat.apply(lambda x: x.replace("Books", ""))
    df.cat = df.cat.apply(lambda x: x.replace("  ,  ,  ", ""))
    df.cat = df.cat.apply(lambda x: x.replace("Kindle Store", ""))
    df.cat = df.cat.apply(lambda x: x.replace("Kindle e", ""))
    df.cat = df.cat.apply(lambda x: x.replace(",  ,  ,  ", ""))
    df.cat = df.cat.apply(lambda x: x.replace(" ()", ""))
    df.cat = df.cat.apply(lambda x: x.replace("  ,  ", ""))
    df.cat = df.cat.apply(lambda x: x.replace(",  ", ","))
    print("cats cleaned")
    return df


# finding isbn searches and kicking them out
def ScrubTerms(df):
    df.search_term = df.search_term.str.replace("books", "")
    df.search_term = df.search_term.str.replace("book", "")
    df.search_term = df.search_term.str.replace("boosks", "")
    df.search_term = df.search_term.str.replace("the", "")
    df.search_term = df.search_term.str.replace("best", "")
    df.search_term = df.search_term.str.replace("sellers", "")
    df.search_term = df.search_term.str.replace("seller", "")
    df.search_term = df.search_term.str.strip()
    df = df[df.search_term != ""]
    df.search_term = remove_punct(df.search_term)
    df = df[~df.search_term.str.contains(r"[0-9]")]
    print("terms scrubbed")
    return df


# clean up punctuation from whatever column applied to
def remove_punct(text):
    text = [(i.lower()) for i in text]
    new_words = []
    for word in text:
        w = re.sub(r"[^\w\s]", "", word)
        w = re.sub(r"\_", "", w)
        w = re.sub(r"\n", "", w)
        new_words.append(w)
    print("punctuation cleared")
    return new_words


# make the categories not ugly so I can use them as filters for finding relevant content
def prep_cats(df):
    df.cat = [(i.split(",")) for i in df.cat]
    df.cat = [list(set(i)) for i in df.cat]
    df.cat = df["cat"].apply(", ".join)
    df.cat = [(i.lower()) for i in df.cat]
    # df.cat = df.cat.str.split(",")
    print("cats prepped")
    return df




# applied the Spacy Neural Network on search terms to intelligently flag Proper Nouns and Person for cleaning.
# used to take search terms with the intent being to remove those that are names, and any futher non-useful terms
def word_tagging(df):
    nlp = spacy.load("en_core_web_lg")
    df["nlpd"] = df.search_term.apply(nlp)
    entities1 = []
    for i in range(len(df)):
        entities1.append(
            [
                (
                    X.is_stop,
                    X.is_alpha,
                    X.shape_,
                    X.pos_,
                    X.dep_,
                    X.ent_type_,
                    X.head.pos_,
                    X.tag_,
                )
                for Y in df["nlpd"][i].ents
                for X in Y
            ]
        )
    df["entities1"] = entities1
    df["entities1"] = [
        (str(i).translate(str.maketrans("", "", string.punctuation)))
        for i in df["entities1"]
    ]
    print("words tagged")
    return df


# identifying where values have hit the top 90 percent in popularity
def FlagTopNinety(df):
    in_top_ninety = []
    for i in range(len(df)):
        if (
            df.avg_searchrank_pct[i]
            >= 85.00  # <- you can adjust the threshold if terms become too restrictive/admissive,
        ):
            in_top_ninety.append("1")
        else:
            in_top_ninety.append("0")

    df["near_ninety"] = in_top_ninety
    df.near_ninety = df.near_ninety.astype(int)
    print("top 90 flagged")
    return df


# below utilizes a pre-trained probabilistic model to identify given/surenames in data.
# this is useful for an additional layer of data cleaning.

# you probably wont need to mess with this, since it's literally just identifying first and last names.
# it's p. static
def ID_RemoveNames(df):
    named_list = []
    for i in range(len(df)):
        named_list.append(pp.parse(df.search_term[i]))

    named_list
    namesdf = pd.DataFrame(named_list)
    namesdf = namesdf.drop(namesdf.iloc[:, 5:24], axis=1)
    namesdf = namesdf.rename(
        columns={0: "name0", 1: "name1", 2: "name2", 3: "name3", 4: "name4"}
    )

    namesdf.fillna("no value", inplace=True)
    namesdf = namesdf.applymap(str)

    gvnnames = namesdf[(namesdf.name0.str.contains("GivenName"))]
    gvnnames = gvnnames.drop_duplicates()
    gvnnames = gvnnames[["name0", "name1"]]
    gvnnames.reset_index(drop=True, inplace=True)

    # new data frame with split value columns
    new = gvnnames["name0"].str.split(",", n=1, expand=True)
    new2 = gvnnames["name1"].str.split(",", n=1, expand=True)

    # # making separate first name column from new data frame
    gvnnames["firstname"] = new[0]
    gvnnames["lastname"] = new2[0]

    # # making separate last name column from new data frame
    gvnnames["name_type1"] = new[1]
    gvnnames["name_type2"] = new2[1]

    # # Dropping old Name columns
    # data.drop(columns =["Name"], inplace = True)

    gvnnames1 = gvnnames[["firstname", "name_type1", "lastname", "name_type2"]]

    gvnnames1[gvnnames1.columns] = gvnnames1.apply(lambda x: x.str.strip("("))
    gvnnames1[gvnnames1.columns] = gvnnames1.apply(lambda x: x.str.strip(")"))
    gvnnames1[gvnnames1.columns] = gvnnames1.apply(lambda x: x.str.strip("''"))

    nlp = spacy.load("en_core_web_lg")
    for col_name in ["firstname", "lastname"]:
        gvnnames1[col_name] = gvnnames1[col_name].apply(nlp)

    entities1 = []
    for i in range(len(gvnnames1)):
        entities1.append(
            [
                (X.pos_, X.ent_type_, X.head.pos_, X.tag_)
                for Y in gvnnames1["firstname"][i].ents
                for X in Y
            ]
        )

    entities2 = []
    for i in range(len(gvnnames1)):
        entities2.append(
            [
                (X.pos_, X.ent_type_, X.head.pos_, X.tag_)
                for Y in gvnnames1["lastname"][i].ents
                for X in Y
            ]
        )

    gvnnames1["entities1"] = entities1
    gvnnames1["entities1"] = [
        (str(i).translate(str.maketrans("", "", string.punctuation)))
        for i in gvnnames1["entities1"]
    ]
    gvnnames1["entities2"] = entities2
    gvnnames1["entities2"] = [
        (str(i).translate(str.maketrans("", "", string.punctuation)))
        for i in gvnnames1["entities2"]
    ]

    gvnnames1 = gvnnames1[
        (gvnnames1.entities1.str.contains("PROPN PERSON PROPN NNP", regex=True))
        | (gvnnames1.entities2.str.contains("PROPN PERSON PROPN NNP", regex=True))
    ]
    gvnnames1 = gvnnames1.applymap(str)

    names = gvnnames1.assign(
        name=gvnnames1[["firstname", "lastname"]].apply(" ".join, axis=1)
    ).drop(
        ["firstname", "lastname", "name_type1", "name_type2", "entities1", "entities2"],
        axis=1,
    )
    names = names.drop_duplicates()
    names = names.name.tolist()
    names = remove_punct(names)
    df = df[(~df.search_term.isin(names))]
    print("id names removed")
    return df


# get rid of names
def FilterPeople(df):
    df = df[~(df.entities1.str.contains("PERSON"))]
    print("people filtered")
    return df


# this functaion evaluates search term and product title to remove matches between those. the goal is to have
# relatively weak matches to avoid instances of direct book title searches


def EvalSequences(df):
    df["seq_score"] = df[["ST_stemmed", "Item_stemmed"]].apply(
        lambda x: textdistance.ratcliff_obershelp(*x), axis=1
    )
    df.seq_score = df.seq_score * 100
    df = df[(df.seq_score <40)]
    df.reset_index(drop=True,inplace=True)
    print("eval sequences done")
    return df


# make a year-month column
def ExtractYear_Month(df):
    df["year_month"] = pd.to_datetime(df["search_date"]).dt.strftime("%Y-%m")
    print("year/month extracted")
    return df


# using saved list of authors and book titles to remove them from data
def Filter_Author_Title(df, title_author):
    title_author.Name = title_author.Name.str.replace("[^\w\s]", "")
    titles = title_author[title_author.Label == "title"]
    authors = title_author[title_author.Label == "author"]
    df = df[
        (~df.search_term2.isin(titles.Name))
        & (~df.search_term2.isin(authors.Name))
    ]
    print("title/author filtered")
    return df



# ugly but on going list focused on removing hard-to-expunge garbage terms

def remove_noise(df):
    exclusion = pd.read_excel('exclusion_list.xlsx')
    df = df[(~df.entities1.str.contains("|".join(exclusion)))]
    to_replace = ["best sellers", "best sellers of all time", "all time"]
    replace_with = ["", "", ""]
    df.search_term = df.search_term.replace(to_replace, replace_with, regex=True)

    df.search_term = [(i.strip()) for i in df.search_term]
    df.search_term.replace("", np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("noise removed")
    return df




# cleans up values in csv of author/books
def clean_titles(title_author):
    title_author.Name = title_author.Name.astype(str)
    title_author.Name = remove_punct(title_author.Name)
    title_author.Name = title_author[~title_author.Name.str.contains(r"[0-9]")]
    title_author.dropna(inplace=True)
    print("titles and authors a cleaned")
    return title_author


# measuring distance from today - a month
def TimeFromToday(df):
    today = pd.to_datetime(date.today())
    df["search_date"] = pd.to_datetime(df["search_date"])
    df["from_today"] = today - df["search_date"]
    df = df[df.from_today <= np.timedelta64(31, "D")]
    print("window narrowed")
    return df


######### compiling data for modeling ##############
def BuildDf(df):
    today = date.today()
    # select main columns
    df = df[
        ["year_month", "search_date", "search_term", "search_rank_pct", ]
    ]
    df = df.drop_duplicates()
    # group and summarize columns
    df2 = (
        df.groupby(["year_month", "search_term"])
        .agg({"search_term": "count", "search_rank_pct": "mean", })
        .rename(
            columns={
                "search_term": "count_term",
                "search_rank_pct": "avg_search_rank_pct",
                "year_month": "yearmonth",
            }
        )
        .reset_index()
    )
    df2.columns = [
        "year_month",
        "search_term",
        "count_term",
        "search_rank_pct",
    
    ]

    # calculate change in counts per month
    df2["count_diff"] = df2.groupby(["search_term"])["count_term"].diff()
    df2.fillna(0, inplace=True)

    # calc monthly percent change
    df2["percent_change"] = (
        df2.groupby("search_term")["count_term"].apply(lambda x: x / x.shift(1) - 1)
        * 100
    ) - 1
    df2.fillna(0, inplace=True)

    # find min and max date per term
    first_val = df2.groupby(["search_term"]).first().reset_index()
    last_val = df2.groupby(["search_term"]).last().reset_index()

    first_val = first_val[
        ["search_term", "year_month", "count_term", "count_diff", "percent_change",]
    ]
    first_val.columns = [
        "search_term",
        "first_searched",
        "count_term",
        "count_diff",
        "percent_change",
    ]
    last_val = last_val[["search_term", "year_month"]]
    last_val.columns = ["search_term", "last_searched"]
    firstlast = pd.merge(last_val, first_val, how="inner", on="search_term")
    firstlast = firstlast[["search_term", "last_searched", "first_searched"]]
    df2 = pd.merge(firstlast, df2, how="inner", on="search_term")
    df2 = df2[
        [
            "year_month",
            "search_term",
            "first_searched",
            "last_searched",
            "count_term",
            "count_diff",
            "percent_change",
            "search_rank_pct",
        ]
    ]
    df2 = df2[df2.search_term != ""]
    # extract length of time between today and min/max search dates
    df2["time_since_first_search"] = pd.to_datetime(today) - pd.to_datetime(
        df2.first_searched
    )
    df2["time_since_last_search"] = pd.to_datetime(today) - pd.to_datetime(
        df2.last_searched
    )
    # find consecutive searches
    df2["consecutive_months"] = df2.groupby("search_term").cumcount() + 1
    mask = df2.time_since_first_search > pd.Timedelta(197, "D")
    mask = mask.astype(np.int)
    # flag older seaches
    df2["older_than_six_months"] = mask
    df2.reset_index(drop=True, inplace=True)
    df2 = df2.rename(columns={"search_rank_pct": "avg_searchrank_pct"})
    # flag top 90 percentile
    df2 = FlagTopNinety(df2)

    # calc magnitude of change
    df2["difference_zscore"] = df2.groupby(["search_term"])["count_diff"].transform(
        stats.zscore
    )
    df2 = df2.fillna(0)
    df2["time_since_first_search"], df2["time_since_last_search"] = (
        df2["time_since_first_search"].dt.days,
        df2["time_since_last_search"].dt.days,
    )
    kwdlist = list(df2.search_term.unique())
    print("data built")
    return df2,kwdlist


#### google trends pulls ######
def meltdf(df):
    df=df.melt(id_vars=["date"], 
        var_name="search_term", 
        value_name="search_volume")
    return df

# make a year-month column
def ExtractYear_Month2(df):
    df["year_month"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
    print("year/month extracted")
    return df

def chunks(items, chunkSize):
        for i in range(0, len(items), chunkSize):
            yield items[i:i+chunkSize]


def pullgtrendsdata(kwdlst):
    dir_path = os.path.dirname(os.path.realpath('pytrends_feature.ipynb'))+'/'
    for chunkIndex, chunk in enumerate(chunks(kwdlst, 5)):
        print('%2d) getting google trends for %s...' % (chunkIndex+1, chunk), end='')
        chunkOutputFile="chunk%02d.csv" % (chunkIndex+1)
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(kw_list=chunk, timeframe='today 3-m', geo='US')
        
        output = io.StringIO()
        interest_over_time_df = pytrends.interest_over_time()
        interest_over_time_df.to_csv(path_or_buf=dir_path + chunkOutputFile)
        print('done, saved to %s' % chunkOutputFile)

    all_files = glob.glob(os.path.join(dir_path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, axis=1)
    concatenated_df_clean = (concatenated_df.drop('isPartial',1))
    concatenated_df_clean = concatenated_df_clean.loc[:,~concatenated_df_clean.columns.duplicated()]
    concatenated_df_clean= meltdf(concatenated_df_clean)
    fileList = glob.glob(dir_path+'*.csv', recursive=True)
    for filePath in fileList:
        try:
            os.remove(filePath)
        except OSError:
            print("Error while deleting file")
    return concatenated_df_clean

def PrepareForModeling(df,googledf):
    scaler = MinMaxScaler()
    df = (
        df.groupby("search_term")
        .agg(
            {
                "count_term": "mean",
                "count_diff": "mean",
                "percent_change": "mean",
                "avg_searchrank_pct": "mean",
                "time_since_first_search": "mean",
                "time_since_last_search": "mean",
                "consecutive_months": "mean",
                "older_than_six_months": "mean",
                "near_ninety": "mean",
                "difference_zscore": "mean",
                
            }
        )
        .reset_index()
    )
    df = word_tagging(df)
    df = remove_noise(df)
    df["scaled_count_term"] = scaler.fit_transform(df[["count_term"]])
    df["scaled_count_diff"] = scaler.fit_transform(df[["count_diff"]])
    df["scaled_percent_change"] = scaler.fit_transform(df[["percent_change"]])
    df["scaled_avg_searchrank_pct"] = scaler.fit_transform(df[["avg_searchrank_pct"]])
    df["scaled_time_since_first_search"] = scaler.fit_transform(
        df[["time_since_first_search"]]
    )
    df["scaled_time_since_last_search"] = scaler.fit_transform(
        df[["time_since_last_search"]]
    )
    df["scaled_consecutive_months"] = scaler.fit_transform(df[["consecutive_months"]])
    df = pd.merge(googledf, df, how="inner", on="search_term")
    print("data ready for model")
    return df


def labelClusters(df):
    conditions = [
        (df["Cluster"] == 0),
        (df["Cluster"] == 1),
        (df["Cluster"] == 2),
        (df["Cluster"] == 3),
    ]
    choices = [
        "new/growingsearches",
        "lowinterest_below90_old",
        "steadylowinterest_below90",
        "highsustainedinterest_typically>=90",
    ]
    df["clust_desc"] = np.select(conditions, choices)
    print("clusters labeled")
    return df


def ApplyKmeans(df):
    kmeans = KMeans(n_clusters=4, random_state=3425, max_iter=100)
    y = kmeans.fit_predict(
        df[
            [
                "scaled_count_term",
                "scaled_count_diff",
                "scaled_percent_change",
                "scaled_avg_searchrank_pct",
                "scaled_time_since_first_search",
                "scaled_time_since_last_search",
                "scaled_consecutive_months",
                "older_than_six_months",
                "near_ninety",
                'google_scaled_count_term', 
                'google_scaled_count_diff',
                'google_scaled_percent_change',
                'google_scaled_difference_zscore'
                
            ]
        ]
    )

    df["Cluster"] = y
    df = labelClusters(df)
#     df = df[(df.Cluster == 3) | (df.Cluster == 0)]
    print("done with kmeans")
    return df


def SelectTopics(df):
    topicslist = []
    for i in range(len(df)):
        if (
            df.Cluster[i] == 0
            and df.near_ninety[i] >= 0.5
            and df.consecutive_months[i] >= 3
            and df.count_term[i] >= np.mean(df.count_term)
        ):
            topicslist.append(df.search_term[i])
        elif (
            df.Cluster[i] == 3
            and df.near_ninety[i] == 1
            and df.consecutive_months[i] >= 3
            and df.time_since_last_search[i] <= 90
            and df.count_term[i] >= np.median(df.count_term)
        ):
            topicslist.append(df.search_term[i])
    print("topics selected")
    return topicslist


#### retrieve searchterms found through modeling from main dataset along with a few additional columns ######
def PullTopicsFromData(terms, df):
    df = df[df["search_term2"].isin([i for i in terms])]
    df = df[["year_month", "search_term", "search_rank_pct", "item", "cat",]]
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("topics pulled")
    return df2


##### additional functions #####
# convert string to list
def Convert(string):
    li = list(string.split(","))
    return li


# expand category columns for evaluation
def SplitOpenData(df):
    df = (
        df["cat"]
        .apply(lambda x: pd.Series(x[0]))
        .stack()
        .reset_index(level=1, drop=True)
        .to_frame("cat")
        .join(df[["search_term"]], how="left")
    )
    return df


# retreive the top categories per term.
def GetTopNodes(df, df2):
    first = df.groupby(["search_term", "cat"]).agg({"cat": "count"})
    second = (
        first.groupby(level=0)
        .apply(lambda x: 100 * x / float(x.sum()))
        .rename(columns={"cat": "cat_count"})
        .reset_index()
    )
    top_2_largest = (
        second.groupby(["search_term"])["cat_count"].nlargest(2).reset_index()
    )
    top_nodes = second[second.index.isin(top_2_largest.level_1)]
    top_nodes = top_nodes.groupby("search_term")["cat"].apply(list).reset_index()
    top_nodes.cat = top_nodes.cat.apply(listToString)
    top_nodes["top_node_1"], top_nodes["top_node_2"] = top_nodes.cat.str.split(
        ",", 1
    ).str
    top_nodes = top_nodes[["search_term", "top_node_1", "top_node_2"]]
    df2 = df2.copy()
    df2 = df2[["year_month", "search_term", "search_rank_pct"]]
    df2 = pd.merge(top_nodes, df2, how="inner", on="search_term").drop_duplicates()
    df2 = (
        df2.groupby(["year_month", "search_term", "top_node_1", "top_node_2"])
        .agg({"search_term": "count", "search_rank_pct": "mean",})
        .rename(
            columns={
                "search_term": "count_search_term",
                "search_rank_pct": "avg_searchrank_pct",
            }
        )
        .reset_index()
    )

    print("dataframe grouped & nodes extracted!")
    return df2


def arrangeGTdata(df):
    df= ExtractYear_Month2(df)
    df = (
        df.groupby(["year_month", "search_term"])
        .agg({"search_volume": "sum" })
        .rename(
            columns={
                "search_volume": "google_sum_search_vol"
            }
        )
        .reset_index()
    )

    df["google_count_diff"] = df.groupby(["search_term"])["google_sum_search_vol"].diff()
    df.fillna(0, inplace=True)

    df["google_percent_change"] = (
    df.groupby("search_term")["google_sum_search_vol"].apply(lambda x: x / x.shift(1) - 1)
            * 100
        ) - 1
    df.fillna(0, inplace=True)
    df["google_difference_zscore"] = df.groupby(["search_term"])["google_count_diff"].transform(
            stats.zscore
        )
    df = df.fillna(0)
    scaler = MinMaxScaler()
    df = (df.groupby(["search_term"])
        .agg(
            {
                "google_sum_search_vol": "mean",
                "google_count_diff": "mean",
                "google_percent_change": "mean",
                "google_difference_zscore": "mean",
                
            }
        ).reset_index())
    df["google_scaled_count_term"] = scaler.fit_transform(df[["google_sum_search_vol"]])
    df["google_scaled_count_diff"] = scaler.fit_transform(df[["google_count_diff"]])
    df["google_scaled_percent_change"] = scaler.fit_transform(df[["google_percent_change"]])
    df["google_scaled_difference_zscore"] = scaler.fit_transform(df[["google_difference_zscore"]])
    return df



def GrabTopics():
    stemmer = SnowballStemmer("english")
    # read in csv for authors and titles
    title_author = pd.read_excel("file.xlsx")
    # clean
    title_author = clean_titles(title_author)
    ### read data in ###
    df = get_data1()
    ### processing ###
    df = ExtractYear_Month(df)
    df = df[df.year_month >= "2019-01"]
    df["item2"] = remove_punct(df.item)
    df["search_term2"] = remove_punct(df.search_term)
    df = Filter_Author_Title(df, title_author)
    df["ST_stemmed"] = df.search_term2.apply(stemmer.stem)
    df["Item_stemmed"] = df.item2.apply(stemmer.stem)
    df2 = ScrubTerms(df)
    df2 = EvalSequences(df2)
    ### model prep and execution ###
    df3 = BuildDf(df2)
    df3 = PrepareForModeling(df3)
    df3 = ApplyKmeans(df3)
    df3.reset_index(drop=True,inplace=True)
    termlist = SelectTopics(df3)
    topicalDF = PullTopicsFromData(termlist, df)
    return termlist,topicalDF
    
    

