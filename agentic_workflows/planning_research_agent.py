from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel
import asyncio
from enum import Enum
import json
import os


load_dotenv()

"""
Workflow:
1. User provides a research topic
2. Orchestrator creates a research plan with subtasks
3. Workers execute subtasks in parallel (when dependencies allow)
4. Synthesizer combines all results into a final output
"""

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class ToolCall(BaseModel):
    tool_name: str
    tool_input: str

class ToolUsage(BaseModel):
    need_tool: bool
    tools: List[ToolCall] = []

class SubTask(BaseModel):
    id: str
    description: str
    dependencies: List[str]
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""

class ResearchPlan(BaseModel):
    subtasks: List[SubTask]

class Tool:
    """Base class for research tools"""
    def run(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement run method")
        
    @property
    def name(self) -> str:
        raise NotImplementedError("Subclasses must implement name property")
        
    @property
    def description(self) -> str:
        raise NotImplementedError("Subclasses must implement description property")

class WebSearchTool(Tool):
    """A tool for web searches using the Exa API"""
    def __init__(self):
        super().__init__()
        self.is_available = False
        try:
            from exa_py import Exa
            api_key = os.getenv("EXA_API_KEY")
            if not api_key:
                raise ValueError("EXA_API_KEY environment variable not found")
            self.client = Exa(api_key=api_key)
            self.is_available = True
        except (ImportError, ValueError) as e:
            print(f"Warning: Web search tool initialization failed: {str(e)}")
            self.client = None
    
    @property
    def name(self) -> str:
        return "web_search"
        
    @property
    def description(self) -> str:
        return "Search for information on the web"
        
    def run(self, query: str) -> str:
        """
        Execute a web search using the Exa API.
        
        Args:
            query (str): The search query
            
        Returns:
            str: A formatted string containing search results with titles,
                 snippets, and source URLs
        """
        if not self.is_available:
            return f"Web search unavailable (Exa API not configured). Query was: {query}"
            
        try:
            # Clean and format the query
            cleaned_query = query.strip()
            
            # Perform the search with Exa
            results = self.client.search_and_contents(
                query=cleaned_query,
                num_results=5,  # Limit to top 5 results
                use_autoprompt=True,  # Use AI to improve the query
                text=True,  # Include text snippets in results
                type="keyword"  # Use keyword search
            )
            
            # Format the results
            formatted_results = []
            for i, result in enumerate(results.results, 1):
                # Extract and clean the text snippet
                snippet = result.text.strip().replace('\n', ' ')
                if len(snippet) > 300:
                    snippet = snippet[:297] + '...'
                
                formatted_result = f"""
                    Result {i}:
                    Title: {result.title}
                    Snippet: {snippet}
                    URL: {result.url}
                    ---
                """
                formatted_results.append(formatted_result)
            
            # Combine all results
            all_results = "\n".join(formatted_results)
            if not all_results.strip():
                return f"No results found for query: {query}"
                
            return f"Search Results for '{query}':\n{all_results}"
            
        except Exception as e:
            return f"Error performing web search for '{query}': {str(e)}"

class Orchestrator:
    """
    The Orchestrator creates and manages the research plan.
    It breaks down complex research tasks into subtasks and
    coordinates their execution.
    """
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def create_research_plan(self, topic: str) -> ResearchPlan:
        """Create a research plan for a given topic"""
        developer_prompt = f"""
            You are a research planner. Create a detailed research plan for 
            the topic provided by the user.
            Break down the research into subtasks. Each subtask should have:
            1. A unique ID
            2. A clear description
            3. Dependencies (IDs of tasks that must be completed first)
            
            Return the plan as a list of subtasks with the following structure.
            Each subtask should be a dictionary with the following keys:
            "id": string,  # Unique identifier for the subtask
            "description": string,  # Description of the subtask
            "dependencies": list[string],  # List of IDs of subtasks that must be completed first
            
            "subtasks": [
                {{
                    "id": string,
                    "description": string,
                    "dependencies": list[string],
                    "tool_requirements": list[string]
                }}
            ]
        """
        input = [
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": topic}
        ]
        
        response = self.client.responses.parse(
            model="gpt-4o-mini",
            input=input,
            text_format=ResearchPlan,
            temperature=0
        )
        
        print(f"\nCreated research plan for topic '{topic}'")
        for task in response.output_parsed.subtasks:
            print(f"\n * Subtask ID: {task.id}, Description: {task.description}, Dependencies: {task.dependencies}")

        return response.output_parsed

class Worker:
    """
    Workers are specialized agents that execute specific research tasks.
    Each worker can use a set of tools to complete their assigned task.
    """
    def __init__(self, tools: Dict[str, Tool]):
        self.client = OpenAI()
        self.tools = tools

    def _get_tools_description(self) -> str:
        """Get a description of available tools"""
        if not self.tools:
            return "No tools available."
        return ", ".join([f"{tool.name}: {tool.description}" for tool in self.tools.values()])

    async def execute_task(self, task: SubTask) -> str:
        """Execute a research subtask"""
        # First, determine how to use the required tools
        planning_prompt = f"""
            You are a research assistant. You need to complete this task:
            {task.description}
            
            You have access to these tools: {self._get_tools_description()}
            
            If you need to use tools to complete the task,
            explain which tools you will use to complete the task.
            Provide tool name and tool input in the format:
            tool_name: <tool name>
            tool_input: <input for the tool>
            For example:
            tool_name: web_search
            tool_input: <search query>
            ...
            
            If no tools are needed, just return "False".

            You do not need to complete the task yet,
            just provide the tools needed, if any.
        """
        input = [
            {"role": "developer", "content": planning_prompt},
            {"role": "user", "content": "Plan the task execution"}
        ]

        response = self.client.responses.parse(
            model="gpt-4o-mini",
            input=input,
            text_format=ToolUsage,
            temperature=0
        )
        
        tool_usage = response.output_parsed

        print(f"\n --> Planning for task '{task.description}':")
        for tool in tool_usage.tools:
            print(f"Tool Name: {tool.tool_name}, Tool Input: {tool.tool_input}")

        tool_results = []
        for tool_call in tool_usage.tools:
            if not isinstance(tool_call, ToolCall):
                return f"Error: Invalid tool call format: {tool_call}"
            tool_name = tool_call.tool_name
            tool_input = tool_call.tool_input
            
            if tool_name not in self.tools:
                return f"Error: Tool '{tool_name}' is not available."
            
            # Check if the tool is available
            tool = self.tools[tool_name]
            tool_results.append(tool.run(tool_input))


        # Synthesize the the output based on the tool results
        synthesis_prompt = f"""
            You are a research assistant. You have to complete this task:
            {task.description}
            
            Using this tool: {self._get_tools_description()}
            
            Here are the results from the tools:
            {tool_results}
            
            Use the results from the tools to complete the task.
        """
        input = [
            {"role": "developer", "content": synthesis_prompt},
            {"role": "user", "content": "Synthesize the results"}
        ]
        
        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=input,
            temperature=0
        )

        return response.output_text        


class Synthesizer:
    """
    The Synthesizer collects results from all completed subtasks
    and combines them into a coherent final output.
    """
    def __init__(self):
        self.client = OpenAI()

    def _get_response(self, input: List[Dict[str, str]], model_name: str="gpt-4o-mini") -> str:
        """Get a response from the OpenAI API"""
        response = self.client.responses.create(
            model=model_name,
            input=input,
            temperature=0
        )
        return response.output_text

    def _format_subtask_results(self, subtasks: List[SubTask]) -> str:
        """Format subtask results for synthesis"""
        formatted_results = []
        for task in subtasks:
            if task.status == TaskStatus.COMPLETED:
                formatted_results.append(f"Subtask ID: {task.id}\nDescription: {task.description}\nResult: {task.result}\n")
            else:
                formatted_results.append(f"Subtask ID: {task.id} is not completed yet.")
        return "\n".join(formatted_results)

    def synthesize_results(self, topic: str, plan: ResearchPlan) -> str:
        """Synthesize all subtask results into a final output"""
        synthesis_prompt = f"""
            You are a research synthesizer. You need to combine the findings
            from the following research subtasks into a coherent final output.
            
            Research Topic: {topic}
            
            Subtask Results:
            {self._format_subtask_results(plan.subtasks)}
            
            Create a well-structured synthesis that:
            1. Introduces the research topic
            2. Presents the key findings
            3. Provides a conclusion
            
            Make the output clear, concise, and well-organized.
        """
        input = [
            {"role": "developer", "content": synthesis_prompt},
            {"role": "user", "content": "Synthesize the research results"}
        ]

        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=input,
            temperature=0
        )
        return response.output_text


class ResearchAgent:
    """
    The main research agent that coordinates the orchestrator,
    workers, and synthesizer.
    """
    def __init__(self):
        self.tools = {
            tool.name: tool for tool in [
                WebSearchTool(),
            ]
        }
        self.orchestrator = Orchestrator()
        self.worker = Worker(self.tools)
        self.synthesizer = Synthesizer()

    async def _execute_subtasks(self, plan: ResearchPlan) -> None:
        """Execute all subtasks in the research plan"""
        # Track completed tasks
        completed = set()
        
        while len(completed) < len(plan.subtasks):
            # Find tasks whose dependencies are satisfied
            tasks = []
            for task in plan.subtasks:
                if task.id not in completed and all(dep in completed for dep in task.dependencies):
                    tasks.append(task)
            
            # Execute eligible tasks in parallel
            results = await asyncio.gather(*[self.worker.execute_task(task) for task in tasks])
            
            # Update task results and mark as completed
            for task, result in zip(tasks, results):
                task.result = result
                task.status = TaskStatus.COMPLETED
                completed.add(task.id)

    async def research(self, topic: str) -> str:
        """
        Execute the full research workflow:
        1. Create a research plan
        2. Execute the plan using workers
        3. Synthesize the results
        """
        # Create the research plan
        plan = self.orchestrator.create_research_plan(topic)
        
        # Execute all subtasks
        await self._execute_subtasks(plan)
        
        # Synthesize the results
        final_output = self.synthesizer.synthesize_results(topic, plan)
        
        return final_output

if __name__ == "__main__":
    async def main():
        agent = ResearchAgent()
        topic = "The impact of artificial intelligence on climate change"
        result = await agent.research(topic)
        
        print("=" * 40)
        print(f"\nResearch Results:\n{result}")

    # Run the async main function
    asyncio.run(main())