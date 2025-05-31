from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class QuestionAnswering:
    """
    workflow:
    1. determine whether the question is a general question or a coding question.
       Route the question to the appropriate model.
    2. if the question is a general question, route it to the general model.
       The general model uses openai's gpt-4o-mini. This is a less expensive model
       and is capable of handling general questions.
    3. if the question is a coding question, route it to the code model.
       The code model uses openai's gpt-4.1. This is a more expensive model
       more capable of handling coding questions.
    """
    def __init__(self):
        self.client = OpenAI()
    
    def _get_response(self, input: list, model_name: str) -> str:
        """
        Get a response from the OpenAI Response API.
        """
        response = self.client.responses.create(
            model=model_name,
            input=input
        )
        return response.output_text
    
    def _router(self, query: str) -> str:
        """
        Determine query type to route the query to the appropriate model.
        """
        developer_prompt = """
            You are an AI assistant that helps users answer questions.
            Your task is to determine whether the question is a general question or a coding question.
            Respond only with "general" or "coding".
            Do not provide any additional information or context.
        """
        input = [
            {
                "role": "developer",
                "content": developer_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ]
        response = self._get_response(input=input, model_name="gpt-4o-mini")
        return response
    
    def _general_model(self, query: str) -> str:
        """
        Answer general questions using the gpt-4o-mini model.
        """
        developer_prompt = """
            You are an AI assistant that helps users answer general questions.
            Your task is to provide a detailed and informative response to the user's question.
            Make sure to include relevant information and context.
        """
        input = [
            {
                "role": "developer",
                "content": developer_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ]
        response = self._get_response(input=input, model_name="gpt-4o-mini")
        return response
    
    def _code_model(self, query: str) -> str:
        """
        Answer coding questions using the gpt-4.1 model.
        """
        developer_prompt = """
            You are an AI assistant that helps users answer coding questions.
            Your task is to provide a detailed and informative response to the user's question.
            Your response should include code snippets, explanations, and relevant information.
        """
        input = [
            {
                "role": "developer",
                "content": developer_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ]
        response = self._get_response(input=input, model_name="gpt-4.1")
        return response
    
    def answer_question(self, query: str, verbose: bool = False) -> str:
        """
        Answer the question by routing it to the appropriate model.
        """
        query_type = self._router(query)

        if verbose:
            print(f"Query Type: {query_type}")
        
        if query_type == "general":
            return self._general_model(query)
        elif query_type == "coding":
            return self._code_model(query)
        else:
            raise ValueError("Invalid query type")
    
if __name__ == "__main__":
    qa = QuestionAnswering()
    
    print("Welcome to the Question Answering!")
    
    while True:
        question = input("\nEnter question, or ':q' to quit: ").strip()
        
        if question.lower() == ':q':
            print("Goodbye!")
            break
            
        if not question:
            print("Please enter a valid question")
            continue
            
        try:
            print("\nProcessing your question...")
            answer = qa.answer_question(question, verbose=True)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

