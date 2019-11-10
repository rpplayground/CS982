# University of Strathclyde -  MSc Artificial Intelligence and Applications
# CS982 - Big Data Technologies
# Assignment
File Created first created 9th October 2019 by Barry Smart.

## Choice Of Dataset
The choice of data set and the objective of exploring global wealth and health trends was inspired by the BBC Four programme “The Joy of Stats - 200 Countries, 200 Years, 4 Minutes” in which Hans Rosling uses animated data visualisation to tell the story of world development from 1810 to 2010:

https://www.youtube.com/watch?v=jbkSRLYSojo

I've chosen to load country level data from the World Bank to explore world development indicators such as:
- Economic indicators such as Gross Development Product (GDP)
- Population and mortallity rates

## Process
To complete the assignment I followed the series of steps illustrated in the below.  The process was not entirely linear as the illustration suggests, exhibiting many iterations or “loops within loops” (Zumel, 2014) in line with normal data science practice:


## Data Flow
The data flowed through the project in accordance with the following diagram:

![Illustration of Data Flow](assignment1\images\DataFlow.png)
 
## Sourcing the Data
A suitable open data was sourced from the World Bank (World Bank, 2019): https://databank.worldbank.org/ 

The databank spans 3 core dimensions as follows:
1.	Country - Data for 217 countries;
2.	Time - Annual data from 1960 to 2018 inclusive (59 years);
3.	World Development Indicators (WDIs) - an extensive range of 1,432 WDIs.  85 were downloaded, but ultimately these were reduced to 20 following analysis of data quality and correlations.
 
The three WDIs featured most extensively in this assignment were:
•	Life Expectancy at Birth, total (years) – the number of years a newborn infant would live if prevailing patterns of mortality at the time of its birth were to stay the same throughout its life;
•	Gross Domestic Product (GDP) per Capita (current US$) – “is a measure of the size and health of a country’s economy” (Bank of England, 2019) in this case is normalised as a “per capita” amount and standardised in US dollars.
•	Population Growth (annual %) – the rate at which population has grown (or declined) in that year as percentage change from the prior year.

Detailed Instructions about how I sourced the data are in the Appendix of the main report.

## Pre-requisites
The environment I have chose to use to complete this assignment is as follows:
- Visual Studio Code - Microsoft's free and open source integrated development environment
- Python - as thew core development environment - extended with a number of core libraries :
    - Arrays, dataframes and general data wrangling:
        - Numpy
        - Pandas
    - Data visualisation:
        - Matplotlib
        - Seaborn
        - Scipy (Dendrogram)
    - Data preparation and machine learning:
        - Sci-Kit Learn – including: scaling, label encoding, clustering algorithms, generation of metrics, linear regression and naïve Bayes.
- Jupyter Notebooks - used to orchestrate the process that I've followed to take the data through the end to end value chain
- Markdown - use of markdown as the means of generating integrated notes within the Jupyter notebooks
- PyTest - used for running unit tests on any bespoke functions developed to support this project

