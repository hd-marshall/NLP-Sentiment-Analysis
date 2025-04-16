# NLP-Sentiment-Analysis

## Introduction

The public's opinion towards companies can be strong, especially for those with financial ties to a company. Their investment can decide whether a company succeeds or heads for a complete crash, therefore gathering an overall sentiment and using this to analyse if such sentiment is reflected in the financial market, was a beneficial investigation.

In this report I explore public sentiment of investors and traders that actively engage with each other about analysing financial trades, corporate actions and economic factors that could impact their portfolios. I was specifically interested in how Trump's presidency has influenced people's confidence or caution in the markets.

This report uses Natural Language Processing (NLP) techniques on Reddit data for the subreddit r/stocks. By analysing these discussions, the aim is to understand sentiment trends, track shifts in investor confidence and uncover insights into how online trends can show themselves in stock market trends. This analysis can enhance our understanding of retail investor behaviour over time.

## Collection Data Structure

Having the comments in their own field as a list a part of submissions, allows the data frame to hold context of what the comment is responding to and allow for understandability to the response and how the public may be perceiving the post.

### Figure 1. Data structure storing API submission and comment data

```json
{
  "posts": [
    {
      "submission_id": "str, (Unique by Reddit)",
      "submission_date": "int, (UNIX Time)",
      "submission_title": "str, (Text created by User)",
      "submission_author": "str, (User)",
      "submission_score": "int, (Upvotes – Downvotes)",
      "submission_upvote_ratio": "float, (Upvotes/ Downvotes)",
      "submission_num_comments": "int, (No. of comments on post)",
      "submission_text": "str, (Text created by User)",
      "submission_comments": [
        {
          "comment_id": "str, (Unique by Reddit)",
          "comment_author": "str, (User)",
          "comment_score": "int, (Upvotes – Downvotes)",
          "comment_text": "str, (Text created by User)",
          "comment_num_replies": "int, (Total numbers of comments on comment)"
        }
      ]
    }
  ]
}
```

## Explorative Data Analysis

The data collected for this report was made on the 1st of April 2025 at 5:45PM. Using the PRAW Python library to access the Reddit API, I collected last month's records of submissions and the comments for each submission within the subreddit r/stocks. 850 submissions were made in the last month and due to data/time restrictions of this research, I reduced the collection of comments to a max of 35 comments from each submission. Resulting in the collection of just under 30,000 text fields.

Initial data exploration shows that most users that have posted in the last month have posted a maximum of 2 times. Where the most amount of submissions from an active user was 15 times. From the 850 submissions made in the last month, there has been only 230 users submitting these submissions.

For the 26,250 comments collected, there have been 4812 unique users that have commented on any post in the last month at an average of 5 comments per active user. The greatest number of comments by a user has been 48.

### Top 15 K-Words

Analysing the most common words found in these combined submissions and comments using tokenisation and eliminating irrelevant words:

- Stock
- Market
- Year
- Company
- Invest
- Trade
- Buy
- Price
- Share
- Time
- Tariff
- Trump
- Month
- Tesla
- Day

### Figure 2. Frequency of most common terms

_[Image placeholder for frequency graph]_

### Figure 3. Distribution of term frequencies

_[Image placeholder for term distribution graph]_

## Unigrams and Bigrams

### Figure 4. Most common single words and two words together

_[Image placeholder for unigrams and bigrams graph]_

## Preprocessing

After completing the initial exploration and storing of the data, it was necessary to complete preprocessing before using models for large amounts of text fields. With the collection of these, there are only 3 types of fields that are natural language, those being submission_title, submission_text and comment_text (Figure 1).

For model efficiency, I first converted all text to lowercase, deleted stop-words/punctuation and removed words that are irrelevant to the algorithms such as words less than 3 characters long. I then tokenised these strings into useable fields whilst stemming suffixes. Next, I Tokenised the words into an array and stemming parts of the word that don't add as much meaning. Finally, I removed all special characters (ASCI values), URLs and non-English words from the dataset. This was to ensure a uniform and meaningful dataset for the model to interpret.

```
Raw Tokens: ['i', 'just', 'want', 'liberation', 'day', 'to', 'be', 'over']
Tokens after stopword/length filtering: ['liberation', 'day']
Final Stemmed Tokens: {'liber', 'day'}
```

Acronyms for finance terms is very popular and industry standard. To avoid removing them in preprocessing, I applied a regular expression that searches and will than skip these terms found in the text corpus, leaving them intact for topic analysis.

```python
acronym_pattern = r'\$?[A-Z]{2,4}'
acronym_pattern_2 = r'[A-Z]{1,2}\&[A-Z]{1,2}'
```

Running the program with and without this additional term shows a clearer understanding of the posts as it begins to now pick up that there are terms like US, AI and ETF. All common topics and terms that are being discussed regularly and before this preprocessing step did not previously appear in the top 25 topics.

After completing this, the original word count was 901,391 which has been cut by a total of 577,545 words. Leaving 323,846 pre-processed tokenised and stemmed words. This will greatly increase computation efficiency of the model with only meaningful words left. Storing the new tokenised fields in a JSON file in same data structure as before but with preprocessed natural language fields.

## Modelling Data Structure

I created a new data structure for all the new data that is generated from the computation of this research, creating a way to understand and represent the data.

### Figure 4. Submission sentiment and topics data structure

```json
{
  "submission_metadata": {
    "submission_id": {
      "date": "submission_id[submission_date], (Unix to Date Time)",
      "time": "submission_id[submission_date], (Unix to Clock Time)",
      "timestamp": "submission_id[submission_date], (UNIX Time)",
      "sentiment": "float, (Sentiment for Submission)",
      "num_comments": "submission_id[submission_num_comments]",
      "topics": "Array[str], (Strings representing terms that are the topics)"
    }
  }
}
```

`submission_id` is stored in the initial collect data structure (Figure 1). It is used as a header and a way to quickly link back and review the holistic information about the submission.

## Method

### Additional Vader Words

Sentiment analysis begins by understanding sets of English words that can discern the overall look of the user towards a topic. Specific words can be used to categorise the tone of the text, whether it is positive, negative or neutral emotional tone. When analysing the text taken from the subreddit, standard sentiment terms fall short of the mark because they miss industry/community specific terms that can be positive or negative. I Introduced these colloquial words used in the online finance space to the sentiment analysis word list. Slang words such as "moon" or "rocket" can portray optimism to a specific topic or stock while "bear" or "dump" can portray a negative sentiment.

| Positive Words | Weight | Negative Words | Weight |
| -------------- | ------ | -------------- | ------ |
| Moon           | 3.0    | Bear           | -2.0   |
| Mooning        | 3.0    | Bearish        | -2.5   |
| Bullish        | 2.5    | Short          | -1.0   |
| Bull           | 2.0    | Puts           | -1.5   |
| Long           | 1.0    | Bagholder      | -2.0   |
| Calls          | 1.5    | Sell off       | -1.5   |
| Hodl           | 1.0    | Dump           | -2.0   |
| Tendies        | 2.0    | Crash          | -3.0   |
| Rocket         | 2.5    | Recession      | -2.5   |
| Rocketship     | 2.5    | Drilling       | -2.0   |
| Yolo           | 1.5    | Tanking        | -2.5   |
| Buy the dip    | 1.5    | Guh            | -2.0   |
| Btd            | 1.5    | Rugpull        | -3.0   |
| Ath            | 2.0    | Fud            | -1.5   |
| Breakout       | 1.5    | Downgrade      | -1.0   |
| Outperform     | 1.5    | Underperform   | -1.5   |
| Upgrade        | 1.0    | Miss           | -1.0   |

```python
sia.lexicon.update(stemmed_lexicon)
```

By incorporating these words into the sentiment analysis model, I have created more contextually aware system that can interpret the sentiment of the community's outlook on stocks, topics and overall discussion, leading to a more accurate insight of sentiment trends.

### Computing Sentiment for Submissions

Using the collected tokenised JSON file to extract a specific submission, I gathered the following natural language text fields: submission_title, submission_text and comment_text (Figure 1). These fields are then analysed individually with the Vader sentiment function, returning a value. These values are then aggerated by adding them to create a total sentiment variable, which is then divided by the total text fields, iterated to generate an average sentiment score for the whole post.

```
s_title = f_sentiment(submission_title)
s_text = f_sentiment(submission_text)
s_comment = f_sentiment(comment_text)
n = number of comments

Sentiment_post = (s_title + s_text + ∑(from i=0 to n)s_comment_i)/(n+2)
```

This generates and creates an average sentiment score for the submission with context of the comment and users' reactions to the submission. Storing this data, it references the submission_id (Figure 3) for the post and creates a new metadata section with the sentiment score stored as Vader's compound score, which is a normalised value between -1 and 1.

### Computing Topic Analysis for Submissions with TF-IDF

I used the submission_title, submission_text and all comment_text associated to a submission. Subsequently, I took all these natural language fields and computed a Term Frequency (TF) for each of the tokenised fields.

```
TF = (occurrences of term)/(terms in document)
```

I then established the overall relevance of the term to the document corpus by creating a tapering off effect for terms that show to often use Inverse Document Frequency (IDF).

```
C ∈ {submission_title, submission_text, comment_text_1, ..., comment_text_n}

IDF = log(|C|/(number of documents that contain term))

Topic Weighting = TF * IDF
```

The higher value calculation for any individual term is than understood to have a higher relevance and meaning to the overall documents. Storing the 9 highest weighted topics back into the data structure as topics (Figure 4).

The graph shows terms identified through TF-IDF analysis, tallying the counts of the most popular terms as topics, which differs significantly from merely counting the most common keywords. While common keyword frequency (k-word) analysis simply tallies raw word occurrences, TF-IDF applies a weighting system.

### Figure 5. Most common topics

_[Image placeholder for common topics graph]_

### Figure 6. Top topics sentiment distribution

_[Image placeholder for topics sentiment distribution graph]_

### Figure 7. Topic word cloud

_[Image placeholder for topic word cloud]_

In Figure 7, the largest words are the most common topics to be found throughout the dataset and the redder the term is, the overall sentiment is worse, and the opposite applies to green terms. Trump and tariff are the most common topics with an overall negative sentiment.

### Computing Topic Analysis for Submissions with LDA

I then generated the topics by using Latent Dirichlet Allocation (LDA) algorithm, it will find statistically significant word clusters of that frequently reoccur. Each document in the corpus is represented as probability distribution over words and these found topic clusters can assign a submission/post to topic segments. The sentiment function and analysis of posts will remain the same but with a more sophisticated topic allocation structure.

### Figure 8. Clustering of top terms of topics

_[Image placeholder for clustering graph]_

### Figure 9. Topic terms correlation network

_[Image placeholder for correlation network graph]_

I used this new topic analysis and the same sentiment analysis that had been conducted and saved with submission_id (Figure 4). I re-ran the data visualisations but, with new allocation of topics for subreddit submissions.

### Figure 10. Sentiment boxplots using LDA topic allocation

_[Image placeholder for sentiment boxplots]_

The single topics of Trump and Tariff are again seen to have an overall negative sentiment, this is comparable to their topic allocation during TF-IDF.

### Figure 11. Terms buy and sell over time compared to potentially contributing factors to their sentiment

_[Image placeholder for terms comparison graph]_

I plotted the topics of Trump, tariff, market and US against the terms of buy and sell to understand the confidence in the market as terms and how they could correlate. The terms Trump and tariff follow a very similar trend, this could be due to the fact they are a popular pairing and are shown together in many of the LDA's clustering. However, the sentiment of buy looks to follow the sentiment of Trump. When opinion and trust for Trump is low so is the people's sentiment to buy into companies and the economy. Trump and the term buy have a correlation of 0.429.

### Figure 12. Plotted z-score normalisation of the S&P and Trump Sentiment

_[Image placeholder for z-score normalisation graph]_

To look closer at market confidence in relation to Trump's sentiment, I have used the Python library Yahoo Finance for historical data on the S&P500 over the duration of the subreddit submissions that have been used. With z-score normalisation to plot them together in a range that can capture the way the values fluctuate. I found a 0.194 correlation between these two variables in the best case.

## Conclusion

The subreddit has a very positive outlook on many of the popular, interesting terms such as invest, dividend, growth, ETF etc. Whereas topics around Trump, tariffs, recession and Elon have a very low sentiment. Suggesting the community members are not happy with Trump and Elon's conduct because of its current effect on the stock market.

If this research were replicated, Researchers should attempt to gather more information and submissions, thus creating a larger dataset for better modelling for LDA. A limitation of this research was that running the model multiple times can yield differing clusters, resulting in different submissions being allocated as different topics. Resulting in differing plot points and graphs. I found that TF-IDF worked the best for understanding the data. Future research should collect data by only allowing submissions that met a minimum number of comments instead of a maximum of 35. This is because many posts that have no comments are found in the dataset to have the maximum and minimum sentiment scores for specific topics. These scores represent themselves as outliers in the dataset mainly, but these ideas may be fringe and potentially do not represent the common ideas of the community.

Conducting this was research was difficult due the large number of topics spoken about in the subreddit and potentially targeting a more focused group or multiple of these could show more reliable overall information for sentiment.
