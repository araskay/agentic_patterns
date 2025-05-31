from openai import OpenAI
import asyncio
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class DevelopmentPlanner:
    """
    workflow:
    1. create a development plan for the requirements based on the following personas:
        - data scientist
        - software engineer
        - product manager
        - UX designer
    2. aggregate the plans from each persona into a single development plan.
    """
    def __init__(self):
        self.client = OpenAI()
        self.personas = [
            "data scientist",
            "software engineer",
            "product manager",
            "UX designer"
        ]
        self.plans = {}

    async def _get_response(self, input: List[Dict[str, str]], model_name: str = "gpt-4o-mini") -> str:
        """
        Get a response from the OpenAI Response API.
        """
        response = self.client.responses.create(
            model=model_name,
            input=input
        )
        return response.output_text
    
    async def _create_development_plan(self, requirements: str, persona: str) -> str:
        """
        Create a development plan for the requirements based on the persona.
        """
        developer_prompt = f"""
            You are an AI assistant that helps a {persona} create development plans.
            You will create a concise development plan for the requirements provided.
            The plan should be concise and only include tasks, timelines, and resources needed.
            Output the plan in markdown format.
        """
        input = [
            {
                "role": "developer",
                "content": developer_prompt
            },
            {
                "role": "user",
                "content": f"Create a development plan for the following requirements: {requirements}"
            }
        ]
        response = await self._get_response(input=input)
        self.plans[persona] = response
        return response
    
    async def _aggregate_plans(self, plans: dict) -> str:
        """
        Aggregate the plans from each persona into a single development plan.
        """
        developer_prompt = """
            You are given development plans from different personas.
            You will combine the plans into a single development plan.
            The aggregated plan should be well-structured and include all tasks, timelines, and resources.
            Output the aggregated plan in markdown format.
        """
        input = [
            {
                "role": "developer",
                "content": developer_prompt
            },
            {
                "role": "user",
                "content": f"{plans}"
            }
        ]
        response = await self._get_response(input=input)
        return response
    
    
    async def generate_plan(self, requirements: str):
        """
        Run the development planning workflow.
        """
        tasks = [self._create_development_plan(requirements, persona) for persona in self.personas]
        await asyncio.gather(*tasks)
        
        aggregated_plan = await self._aggregate_plans(self.plans)
        return aggregated_plan
    
if __name__ == "__main__":
    async def main(requirements: str):
        planner = DevelopmentPlanner()

        aggregated_plan = await planner.generate_plan(requirements)

        print("Development Plan for Requirements:")
        for persona, plan in planner.plans.items():
            print(f"\n{persona} Plan:")
            print(plan)
        
        print("\n" + "=" * 80 + "\n")
        print("Aggregated Development Plan:")
        print(aggregated_plan)

    # Run the async main function
    requirements = """
        Build an AI agent that can create SQL queries from natural language.
        The agent should be able to understand user queries in the context of their data,
        which may include tables, columns, and relationships, stored in a database.
        The agent should generate SQL queries,
        and allow the user to execute them against the database.
    """    
    asyncio.run(main(requirements=requirements))