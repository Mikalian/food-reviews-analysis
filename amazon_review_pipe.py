
"""
ETL Pipeline: Amazon Fine Food Reviews Enrichment
Extracts unstructured review text and transforms it into structured categorical data
using LangChain and Gemini.
"""

import typing
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import pandas as pd


class AmazonReviewEnrichment(BaseModel):
    """
        Target schema for LLM extraction.
        Field descriptions act as prompt instructions for the LLM to ensure strict categorization.
    """
    Issue_Category: typing.Literal[
        "Bad_Taste",
        "Stale_Expired",
        "Packaging_Issue",
        "Price_Complaint",
        "Other_Issue",
        "None"
    ] = Field(
        description="""Categorize the main complaint. 
        If the review is generally negative but doesn't fit specific categories, use 'Other_Issue'.
        If the review is purely positive or has no complaints, strictly use 'None'."""
    )

    Repurchase_Intent: typing.Literal["Yes", "No", "Unclear"] = Field(
        description="""Would the customer buy this again? If they love it, it's 'Yes'.
        If they say they will throw it away or never buy again, it's 'No'. 
        Otherwise 'Unclear'."""
    )

    Emotional_Tone: typing.Literal[
        "Sarcasm",
        "Frustration",
        "Disappointment",
        "None"
    ] = Field(
        description="""Identify the negative emotional undertone.
                    'Sarcasm' for mocking praise.
                    'Frustration' for annoyance or irritation (e.g., 'what a waste of money').
                    'Disappointment' for unmet expectations. Use 'None' if the review is neutral, factual, or happy."""
    )


def main():
    load_dotenv()

    # Load dataset and sample for POC to manage API costs and execution time.
    # Note: Full dataset processing requires transitioning to batch-processing architecture.
    reviews = pd.read_csv("Reviews.csv")
    sample = reviews.sample(n=250)

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
    parser = PydanticOutputParser(pydantic_object=AmazonReviewEnrichment)
    format_instruction = parser.get_format_instructions()
    # Construct the extraction chain: Prompt -> LLM -> Pydantic Parser
    template_text = """Analyze the following review: {review_text}.
    Based on the review, create an output strictly according to {format_instruction}
    """
    prompt = ChatPromptTemplate.from_template(template_text)
    chain = prompt | llm | parser

    enriched_data = []
    # Process sequentially. Separating metadata (IDs) from payload (Text) to optimize token usage.
    for index, row in sample.iterrows():
        review_id = row["Id"]
        product_id = row["ProductId"]
        review_text = row["Text"]
        score = row["Score"]
        try:
            response = chain.invoke({
                "review_text": review_text,
                "format_instruction": format_instruction,
            })
            # Recombine metadata with the structured LLM output
            enriched_data.append({
                "Id": review_id,
                "ProductId": product_id,
                "Score": score,
                "Issue_Category": response.Issue_Category,
                "Repurchase_Intent": response.Repurchase_Intent,
                "Emotional_Tone": response.Emotional_Tone,
            })
            print(f"Successfully processed ID: {review_id}")
        except Exception as e:
            # Fallback for LLM parsing failures or API timeouts
            print(f"Model failed to process ID: {review_id}. Error: {e}")

    # Persist the transformed data. Using index=False to prevent artifact index columns on reload.
    output_filename = "enriched_amazon_reviews.csv"
    pd.DataFrame(enriched_data).to_csv(output_filename, index=False)
    print("Data successfully saved to 'enriched_reviews_sample.csv' in the current directory.")


if __name__ == "__main__":
    main()
