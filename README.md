# Find-trending-keywords
this is project I worked on  a job where I was set loose to find a way to extract interesting keywords from internal amazon search data. The intent behind the project is to identify terms that can assist in the process for publishing new content. I leverage daily-level amazon search data, as well as google trends data to help identify things like: 

- the consistency of the trend
- popularity 
- sustained popularity 
- gradient 
- direction 
- age of the trend 


Ultimately I comprised various features to describe and capture the behavior of these terms over the course of a year ( and as they come in each month ( when the algorithm is run)) and apply clustering to these features in order to classify the similarities. For example: 

cluster 0 might be things that people searched for, which made it into amazon's search rank ( enough to show up as 1 search), but never reached any ascending levels of popularity, as well as being very old, and unsearched for over 4 months. 

on the other hand, cluster 2 might be those that are very 'trendy', and have been search conssitently over the year, with particularly increase in the last couple of months. 

while cluster 3 might be all new and growing searches which would be tagged on a watch list of sorts. 

I know it might be odd to apply clustering to a 'trend' analysis, but this was not a traditional trend analysis, as much as a selection process. And I needed a way to break down/condense numerous terms with varying search behaviours over time so that it would be more obvious which terms I should actually look into. It isn't represented in the code, but once I find a selection of terms, I perform a forecasting of sorts at the daily level to evaluate their projected trend using some additional features I created. 


With these terms that I find and finally isolate, i evaluate the product information that was purchased using those terms, then filter out a much larger selection of data to find more niche secondary terms, which you may interpret as getting to the bottom of what users "meant to search". This has no casuation basis, mind you. It's simply trying to see what other terms in volume lead to the same purchses. the database is omitted in the code, for obvious reasons. but the code itself is open to you! I hope it can be helpful to anyone trying to find any trends in their own data but face a bunch of uncertainty about how to go about homing into some signals. 
