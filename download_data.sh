#!/bin/bash
if ! [ -d "./res" ]; then
	mkdir res
fi
cd res
if ! [ -f "./Sentiment\ Analysis\ Dataset.csv"]; then
	wget http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip
	unzip Sentiment-Analysis-Dataset.zip
	rm -f Sentiment-Analysis-Dataset.zip
fi
cd ..
