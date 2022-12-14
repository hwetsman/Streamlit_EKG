# Purpose

This project aims to make easier the viewing and categorizing 30-second EKGs from Apple Watch data.

# Current Functionality

There are currently a few functions enabled. When the program runs on new data, that data must be encorporated into an indexing system. So the first function is 'Reset EKG Database.' Once the database is established, the EKGs are assessed for the presence of Premature Atrial Contractions (PACs) and the number of them in each EKG is added to the database. This is accomplished in the 'Show PACs Over Time' function. Finally, one can view the EKGs in the 'Show an EKG' function. 

## Reset EKG Database
There are no choices in this function, and it runs fairly quickly. There are currently no outputs except that a dataframe of the listed EKGs is printed to the terminal at the end of the run.

## Show PACs Over Time
There are also no choices in this function, but it takes longer to run the first time than the first function. Each EKG is analyzed in several steps to assess for PACs. This is the time consuming factor. The output of the function is a timeseries graph showing vertical red lines on any days when an EKG met criteria for Atrial Fibrillation (afib) and, initially, a bar graph of the maximum number of PACs seen each day. The user may choose to see the maxium PACs as a rolling average line graph instead of a bar graph, and, if so, they may choose the number of days over which to compute the rolling average. 

The second graph shown is the average number of PACs per EKG done on each day of the week. It is calculated by taking the total number of PACs seen on EKGs from that day of the week divided by the number of EKGs done that day of the week. The occurrance of Afib line graph notes the number of times Afib has occured on that day of the week

The third graph is the maximum number of PACs in a 30 second EKG seen on EKGs done in each hour of the day. It is calculated by grouping all the EKGs by hour of the day done and selecting the EKG with the maximum number of counted PACs. The red line is the number of times an EKG classified as having Afib occurred during that hour of the day. 

## Show an EKG
The user may choose the year of interest and the month as well as the classification of EKG by Apple's categorization (Sinus Rhythm, Fast Heart Rate, Slow Heart Rate, Atrial Fibrillation, Inconclusive). Finally, there is a dropdown list by day of the EKGs available that fit the user's choices. The drop down includes the information of whether the EKG could be assessed for PACs and, if so, how many PACs are on the EKG. 
