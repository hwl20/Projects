# Project 2
# Goal here is to understand the most popular words used in 1) reddit and 2) twitter posts to discern 
# whether they are positive or negative words according to the sentiments of tidytext package
# Assumptions: We will classify into positive and negative sentiments only (We will not assume netural results here)
# Data is retrieved from: https://www.kaggle.com/cosmos98/twitter-and-reddit-sentimental-analysis-dataset?select=Twitter_Data.csv

getwd()
setwd("../OneDrive - National University of Singapore/NUS Y1/NUS Y1 Summer Break/Projects/R")
load("Sentiment2.RData")
install.packages("tidytext")
install.packages("janeaustenr")
install.packages("stringr")
install.packages("tidyr")
install.packages("ggplot2")
install.packages("reshape2")
install.packages("wordcloud")
install.packages("dplyr")

library(tidytext)
library(janeaustenr)
library(stringr)
library(tidyr)
library(ggplot2)
library(reshape2)
library(wordcloud)
library(dplyr)

############################################################################################################################################################
# Reddit
# Extracting the data
df <- read.csv("Reddit_Data.csv")
names(df)
head(df,2)
df1 <- df
df1 <- df1 %>%
  select(clean_comment)%>%
  mutate(linenumber = row_number())%>%
  unnest_tokens(word,clean_comment)

# Discerning are majority positive or negative sentiment words
bing = get_sentiments("bing")
df10 <- df1 %>%
  inner_join(bing)%>%
  count(index = linenumber%/%10, sentiment)%>%
  spread(key=sentiment, value=n,fill=0)%>%        # fill here is to inform R to fill up missing values (both NA or row doesn't exist covered by fill)
  mutate(sentiment = positive-negative)
df100<-df1 %>%
  inner_join(bing)%>%
  count(index = linenumber%/%100, sentiment)%>%
  spread(key=sentiment, value=n)%>%
  mutate(sentiment = positive-negative)
df200<-df1 %>%
  inner_join(bing)%>%
  count(index = linenumber%/%200, sentiment)%>%
  spread(key=sentiment, value=n)%>%
  mutate(sentiment = positive-negative)
df400<-df1 %>%
  inner_join(bing)%>%
  count(index = linenumber%/%400, sentiment)%>%
  spread(key=sentiment, value=n)%>%
  mutate(sentiment = positive-negative)

# Visualization
ggplot(df10, aes(index, sentiment, fill = "statements"))+
  geom_bar(stat="identity",show.legend = TRUE)
ggplot(df100, aes(index, sentiment, fill = "statements"))+
  geom_bar(stat="identity",show.legend = TRUE)
ggplot(df200, aes(index, sentiment, fill = "statements"))+
  geom_bar(stat="identity",show.legend = TRUE)
ggplot(df400, aes(index, sentiment, fill = "statements"))+
  geom_bar(stat="identity",show.legend = TRUE)

# From the visualizations we can tell that statements positive and negative sentiments are relatively 
# equal, but there seem to be particular group of sentences whereby the sentiments of words used are 
# largely positive



# Top words used in the positive and negative sentiments respectively
df2 <- df1 %>%
  inner_join(bing)%>%
  count(word,sentiment,sort = TRUE) %>%
  filter(n>200)%>%
  mutate(n=ifelse(sentiment=="positive", n, -n))%>%
  mutate(word= reorder(word,n))

df2 %>% ggplot(aes(word, n, fill = sentiment))+
          geom_col()+
          coord_flip()+
          labs(y="Sentiment Score")
# Indeed, the the proportion of positive sentiments words "like","good","right" seems to dominate the 
# words that were used, which results in positive sentiments being more prominent than negative ones


# Word Cloud
df3<- df1 %>%
      inner_join(bing)%>%
      count(word,sentiment,sort = TRUE) %>%
      filter(n>100)%>%
      mutate(n=ifelse(sentiment=="positive", n, -n))%>%
      mutate(word= reorder(word,n))
df3 %>%
  acast(word~sentiment, value.var = "n", fill=0)%>%
  comparison.cloud(colors=c("red","green"), max.words=100)

# Seems like top words are Positive: like,good,well,best,right,free,best,better,work
#                          Negative: hate,shit,fuck,wrong,problem,issue,fucking

# Conclusion: Reddit seems to be a conducive community with much more positive words than negative


############################################################################################################################################################
# Twitter
# Extracting the data
dir()
twitter <- read.csv("Twitter_Data.csv")
twitter_1 <- twitter %>%
                mutate(linenumber = row_number())%>%
                select(-category)%>%
                unnest_tokens(word, clean_text)
bing <- get_sentiments("bing")

# Combining the 2 tables to include positive or negative sentiment labels to each word 
# and group the sentences together
twitter_test <- twitter_1 %>%
              inner_join(bing)%>%
              count(index = linenumber, sentiment)%>%
              spread(key = sentiment, value = n, fill = 0)%>%
              mutate(sentiment = positive - negative)

twitter_100 <- twitter_1 %>%
  inner_join(bing)%>%
  count(index = linenumber%/%100, sentiment)%>%
  spread(key = sentiment, value = n, fill = 0)%>%
  mutate(sentiment = positive - negative)
twitter_200 <- twitter_1 %>%
  inner_join(bing)%>%
  count(index = linenumber%/%200, sentiment)%>%
  spread(key = sentiment, value = n, fill = 0)%>%
  mutate(sentiment = positive - negative)
twitter_500 <- twitter_1 %>%
  inner_join(bing)%>%
  count(index = linenumber%/%500, sentiment)%>%
  spread(key = sentiment, value = n, fill = 0)%>%
  mutate(sentiment = positive - negative)
twitter_1000 <- twitter_1 %>%
  inner_join(bing)%>%
  count(index = linenumber%/%1000, sentiment)%>%
  spread(key = sentiment, value = n, fill = 0)%>%
  mutate(sentiment = positive - negative)

# Plotting to see the sentiment of sentences in group
ggplot(twitter_100, aes(index, sentiment, fill = "Sentiment"))+
  geom_bar(stat = "identity")
ggplot(twitter_200, aes(index, sentiment, fill = "Sentiment"))+
  geom_bar(stat = "identity")
ggplot(twitter_500, aes(index, sentiment, fill = "Sentiment"))+
  geom_bar(stat = "identity")
ggplot(twitter_1000, aes(index, sentiment, fill = "Sentiment"))+
  geom_bar(stat = "identity")
# Seemingly each group seems to contain statements that are generally more positive than negative in nature

# Top Negative and Positive Sentiments words
twitter_2 <- twitter_1%>%
              inner_join(bing)%>%
              count(word,sentiment,sort=TRUE)%>%
              #filter(n>750)%>%
              mutate(n = ifelse(sentiment=="positive", n, -n))%>%
              mutate(word = reorder(word,n))
twitter_2 %>% filter(abs(n)>750)%>%
  ggplot(aes(word,n, fill=sentiment))+
  geom_col()+
  coord_flip()+
  labs(y="Word Sentiment")

# Seems like more positive sentiments by analyzing words from twitter tweets on twitter  

# Word Cloud for twitter
twitter_3 <- 
  twitter_1 %>%
  inner_join(bing)%>%
  count(word,sentiment,sort=TRUE)
twitter_3 %>%
  acast(word~sentiment,value.var = "n",fill = 0)%>%
  comparison.cloud(colors = c("red","green"),max.words=100)

# like is the most popular word amongst all twitter posts, and it is associated as a positive word
save.image("Sentiment2.RData")
