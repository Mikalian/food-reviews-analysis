# Food Reviews Analysis

## Overview
This project analyzes customer reviews to figure out what actually makes customers leave and what triggers negative emotions like frustration and sarcasm. Instead of just looking at average star ratings, this analysis digs into the specific issues behind the scores to provide actionable business insights.

## The Dataset
This project is based on a massive real-world food product review dataset from [Amazon Fine Food Reviews on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). 

Because the original dataset contains hundreds of thousands of rows, running an LLM API on the entire set would be highly inefficient and hit rate limits. Instead, a representative sample of 250 rows was extracted. This sample underwent an AI-driven **data enrichment** process to extract structured features:

* **Issue_Category:** The primary type of problem extracted from the text (e.g., Bad Taste, Stale/Expired, Packaging Issue).
* **Emotional_Tone:** The underlying sentiment of the review (Disappointment, Frustration, Sarcasm, etc.).
* **Score:** The star rating given by the customer (1 to 5).
* **Repurchase_Intent:** Whether the customer plans to buy from us again (Yes, No, Unclear).

## Tech Stack
* **Python:** The core language used for scripting and analysis.
* **Langchain & Google Gemini API:** Applied for automated data enrichment (extracting sentiment, categories, and intent from raw text reviews).
* **Pandas:** Data manipulation, cleaning, and cross-tabulation.
* **Seaborn & Matplotlib:** Data visualization (Heatmaps, Stacked and Grouped Bar Charts).
* **Jupyter Notebook:** Interactive development and presentation of the analytical workflow.

## Key Business Questions Explored
1. **How do different product issues impact customer retention?** 
   Identifying which specific defects are deal-breakers that drive customers away.
2. **Which product issues trigger the most negative emotions?** 
   Using a heatmap to pinpoint exactly which product failures cause the highest levels of frustration and sarcasm.
3. **At what rating do we actually lose the customer?** 
   Investigating the "grey area" of 2-star and 3-star reviews to find the true threshold where customers decide not to return.

## How to View
Simply open the `customer_sentiment_analysis.ipynb` file in this repository. GitHub natively renders Jupyter Notebooks, so you can view all the code and charts directly in your browser without downloading anything.
## Future Improvements
* **API Batching:** Currently, the enrichment script processes rows sequentially. Future optimization will involve batching multiple reviews into a single API prompt to reduce the number of API calls, avoid `429 RESOURCE_EXHAUSTED` rate limits, and speed up the data pipeline.
* **Error Handling & Retries:** Implementing automated retries and error handling to smoothly bypass temporary API quota limits.

## How to Run Locally
If you want to run the notebook and generate the data yourself:
1. Clone this repository.
2. Install the required libraries: `pip install pandas seaborn matplotlib langchain langchain-google-genai`
3. Get a free Google Gemini API key from Google AI Studio.
4. Set your API key as an environment variable or define it in your notebook environment to run the enrichment script.
