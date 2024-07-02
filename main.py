import prompts
from searcher import DenseSearcher
from evaluator import PrometheusEval
from openai import AzureOpenAI
import os

def main():
    dense_searcher = DenseSearcher(collection_name="skt_embedding_e5large")
    #dense_searcher = DenseSearcher(collection_name="skt_embedding_m3dense")
    instruction, response, context = dense_searcher.ask_llm(query="5G 요금제를 알려줘")

    print("Query:", instruction)
    print("Response:", response)
    print("Context:", context)
    model = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2023-12-01-preview"
    )

    # Parameters
    grading_format = "absolute"  # Set to "relative" if needed
    include_reference = True    # Set to True if reference answer should be included
    include_context = True
    criteria = "helpfulness"

    # Initialize the judge with the selected prompt template
    judge = PrometheusEval(
        model=model,
        grading_format=grading_format,
        include_reference=include_reference,
        include_context=include_context
    )

    # Format the score rubric
    #custom_score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)
    score_rubric = prompts.load_rubric(criteria, grading_format)

    # Perform the grading
    feedback, score = judge.single_absolute_grade(
        instruction=instruction,
        response=response,
        rubric=score_rubric,
        context=context,
    )

    # Output the results
    print("Feedback:", feedback)
    print("Score:", score)

if __name__ == "__main__":
    main()

