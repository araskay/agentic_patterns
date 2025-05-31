from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

class EssayWriter:
    """
    workflow:
    1. create an outline
    2. expand the outline by adding content to each section
    3. write an essay based on the expanded outline
    """
    def __init__(self):
        self.client = OpenAI()

    def _get_response(self, input: List[Dict[str, str]], model_name: str="gpt-4o-mini") -> str:
        """
        Get a response from the OpenAI Response API.
        """
        response = self.client.responses.create(
            model=model_name,
            input=input
        )
        return response.output_text
    
    def _create_outline(self, topic: str) -> str:
        """
        Create an outline for the essay.
        """
        developer_prompt = """
            You are an AI assistant that helps users write essays.
            You will create an outline for the essay based on the topic provided.
            The outline should be formatted as a list of sections and subsections
            marked with numbers (1., 2., 3., etc.).
            Include a short (1-2 sentence) description of each section.
            Output the outline in markdown format.
        """
        input = [
            {
                "role": "developer",
                "content": developer_prompt
            },
            {
                "role": "user",
                "content": f"Create an outline for an essay on the topic: {topic}"
            }
        ]
        response = self._get_response(input=input)
        return response

    def _expand_outline(self, outline: str) -> str:
        """
        Expand the outline by adding content to each section.
        """
        developer_prompt = """
            You are an AI assistant that helps users write essays.
            You are provided with an outline, which will expand by adding content to each section.
            The content should be relevant to the topic and well-structured.
            Output the expanded outline in markdown format.
        """
        input = [
            {
                "role": "developer",
                "content": developer_prompt
            },
            {
                "role": "user",
                "content": f"Expand the following outline: ```{outline}```"
            }
        ]
        response = self._get_response(input=input)
        return response
    
    def _write_essay_based_on_expanded_outline(self, expanded_outline: str) -> str:
        """
        Write an essay based on the expanded outline.
        """
        developer_prompt = """
            You are an AI assistant that helps users write essays.
            You are provided with an expanded outline,
            which you will use to write an essay based on it.
            The essay should be coherent and NOT in bullet-point format.
            Output the essay in markdown format.
        """
        input = [
            {
                "role": "developer",
                "content": developer_prompt
            },
            {
                "role": "user",
                "content": f"Write an essay based on the following expanded outline: ```{expanded_outline}```"
            }
        ]
        response = self._get_response(input=input)
        return response
    
    def write_essay(self, topic: str, verbose: bool = False) -> str:
        """
        Use prompt chaining workflow to write an essay based on the topic provided.
        """
        outline = self._create_outline(topic=topic)
        if verbose:
            print(f"\nOutline:\n {outline}")
        expanded_outline = self._expand_outline(outline=outline)
        if verbose:
            print(f"\nExpanded Outline:\n {expanded_outline}")
        essay = self._write_essay_based_on_expanded_outline(expanded_outline=expanded_outline)
        return essay


if __name__ == "__main__":
    writer = EssayWriter()
    
    print("Welcome to the Essay Writer!")
    print("Enter a topic to generate an essay, e.g. 'Accelerated magnetic resonance imaging'.")
    print("Type ':q' to quit the program.")
    
    while True:
        topic = input("\nEnter topic, or ':q' to quit: ").strip()
        
        if topic.lower() == ':q':
            print("Goodbye!")
            break
            
        if not topic:
            print("Please enter a valid topic")
            continue
            
        try:
            print("\nGenerating essay... Please wait...")
            essay = writer.write_essay(topic=topic, verbose=True)
            print("\nEssay:\n")
            print(essay)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

