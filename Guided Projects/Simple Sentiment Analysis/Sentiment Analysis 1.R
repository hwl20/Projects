# Project 1
# Idea is to analyze words relating to positive VS negative sentiments from austen_books() in package janeaustenr
# Project was inspired by code from DataFlair website - https://data-flair.training/blogs/data-science-projects-code/
setwd("C:/Users/ASUS-PC/OneDrive - National University of Singapore/NUS Y1 Summer Break/Projects/R")
install.packages("tidytext")
install.packages("janeaustenr")
install.packages("stringr")
install.packages("tidyr")
install.packages("ggplot2")

# Getting the sentiment indicators and Retrieving the dataset
library(tidytext)
library(janeaustenr)
library(stringr)
library(tidyr)
library(dplyr)
sentiments
get_sentiments("bing")

# Converting textbook texts into a tidy format
austen_books()
which(str_detect(austen_books()$text, "^chapter [\\divxlc]"))
sum(str_detect(austen_books()$text, "^chapter [\\divxlc]"))
tidy_data <- austen_books()%>%
  group_by(book) %>%
  mutate(linenumber = row_number(),
         chapter = cumsum(str_detect(text, regex("^chapter [\\divxlc]", ignore_case = TRUE))))%>%
  ungroup() %>%
  unnest_tokens(word,text)
tidy_data

# Aggregating the positive sentiments
positive_sentiment <- get_sentiments("bing")%>%
  filter(sentiment == "positive")
tidy_data %>%
  filter(book=="Emma")%>%
  semi_join(positive_sentiment) %>%
  count(word,sort=TRUE)

bing <- get_sentiments("bing")
# Joining the tables
Emma_sentiment <- tidy_data%>%
  inner_join(bing)%>%
  count(book = "Emma",index = linenumber%/%80, sentiment)%>%
  spread(key=sentiment, value=n)%>%
  mutate(sentiment = positive - negative)
#unique((tidy_data%>%
#  inner_join(bing)%>%data.frame()%>%filter(book=="Emma"))[,"linenumber"])


# Visualization 
library(ggplot2)

ggplot(Emma_sentiment, aes(index, sentiment, fill = book))+
  geom_bar(stat="identity",show.legend = TRUE)+
  facet_wrap(~book,ncol = 2, scales = "free_x")


counting_words <- tidy_data %>%
  inner_join(bing)%>%
  count(word,sentiment,sort=TRUE)
counting_words

#Visualization 2
counting_words %>%
  filter(n>150)%>%
  mutate(n=ifelse(sentiment=="negative", -n, n))%>%
  mutate(word = reorder(word,n))%>%
  ggplot(aes(word,n, fill = sentiment))+
  geom_col()+
  coord_flip()+
  labs(y="Sentiment Score")
# We observe the most used positive sentiments words and the most used negative sentiments words

# Creation of a word cloud visualization
install.packages("reshape2")
install.packages("wordcloud")
library(reshape2)
library(wordcloud)
counting_words %>%
  acast(word~sentiment, value.var="n",fill=0)%>%
  comparison.cloud(colors = c("red","green"), max.words = 100)
#Word cloud clearly shows the positive and negative sentiments most common words
