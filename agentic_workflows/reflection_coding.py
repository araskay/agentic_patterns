from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class Feedback(BaseModel):
    is_correct: bool
    feedback: str

class Coding:
    """
    workflow:
    1. write code based on the given prompt.
    2. examine the code for correctness and efficiency.
       provide feedback on the code.
    3. if the code is correct, return the code.
       if the code needs improvement, include feedback with the prompt
       and go back to step 1.
    """

    def __init__(self):
        self.client = OpenAI()
    
    def write_code(self, prompt: str) -> str:
        """
        Generate code based on a prompt.

        Args:
            prompt: The programming task description

        Returns:
            str: The generated code
        """
        developer_prompt = """
            You are a skilled programmer. Write code based on the given prompt.
            If feedback is provided, use it to improve the code.
            Provide the complete code without any additional explanations.
        """
        input = [
            {
                "role": "developer",
                "content": developer_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=input,
            temperature=0
        )
        return response.output_text

    def examine_code(self, prompt: str, code: str) -> Feedback:
        """
        Examine code for correctness and efficiency.

        Args:
            prompt: The original programming task
            code: The code to examine

        Returns:
            Feedback: Object containing correctness status and feedback
        """
        developer_prompt = """
            You are a code reviewer. Examine the code for correctness and efficiency.
            Make sure the code meets the following criteria:
            1. It should be syntactically correct and runnable.
            2. It should solve the problem described in the prompt.
            3. It should be efficient and follow best practices.
            4. It should have type annotations and docstrings where appropriate.
            Return 'CORRECT' if the code is good,
            or provide specific feedback for improvement.
            The code was written based on the following prompt:
            {prompt}
        """
        input = [
            {
                "role": "developer",
                "content": developer_prompt
            },
            {
                "role": "user",
                "content": f"Review this code:\n{code}"
            }
        ]
        response = self.client.responses.parse(
            model="gpt-4o-mini",
            input=input,
            text_format=Feedback,
            temperature=0
        )
        return response.output_parsed

    def generate_code(self, prompt: str, max_iterations: int = 3, verbose: bool = False) -> str:
        """
        Generate code with iterative feedback and improvements.

        Args:
            prompt: The programming task description
            max_iterations: Maximum number of improvement attempts
            verbose: Whether to print debug information

        Returns:
            str: The final generated code
        """
        current_prompt = prompt
        
        for i in range(max_iterations):
            if verbose:
                print("=" * 20)
                print(f"Iteration {i + 1} of {max_iterations}")
                #print(f"Current prompt: {current_prompt}")
            code = self.write_code(current_prompt)
            feedback = self.examine_code(prompt, code)
            if verbose:
                print(f"Code generated:\n{code}")
                print(f"Feedback received: {feedback.feedback}")
                print(f"Is the code correct? {'Yes' if feedback.is_correct else 'No'}")
            
            if feedback.is_correct:
                return code
                
            current_prompt = f"""
            Original prompt: {prompt}
            Previous attempt: {code}
            Feedback: {feedback.feedback}
            """
        
        return code  # Return last attempt if max iterations reached

if __name__ == "__main__":
    coding_workflow = Coding()
    initial_prompt = "Write a python function that calculates the factorial of a number."
    final_code = coding_workflow.generate_code(initial_prompt, verbose=True)
    print("*" * 20)
    print("Final Code:\n", final_code)