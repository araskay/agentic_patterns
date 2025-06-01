from dotenv import load_dotenv
import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy import text, create_engine, Engine, inspect
from openai import OpenAI

load_dotenv()

# max iterations for the agent
MAX_ITERATIONS = 30

class SQLTool:
    """Base class for SQL tools"""
    def __init__(self, engine: Engine):
        self.engine = engine
        
    def run(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement run method")
        
    @property
    def name(self) -> str:
        raise NotImplementedError("Subclasses must implement name property")
        
    @property
    def description(self) -> str:
        raise NotImplementedError("Subclasses must implement description property")

class ListTablesTool(SQLTool):
    """Tool for listing tables in the database"""
    @property
    def name(self) -> str:
        return "list_tables"
        
    @property
    def description(self) -> str:
        return "List all tables in the database"
        
    def run(self) -> str:
        with self.engine.connect() as conn:
            # This works for MySQL-compatible databases
            result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result]
            return str(tables)


class GetTableSchemaTool(SQLTool):
    """Tool for getting the schema of a table"""
    @property
    def name(self) -> str:
        return "get_table_schema"
        
    @property
    def description(self) -> str:
        return "Get the schema of a specific table. Usage: get_table_schema(table_name)"
        
    def run(self, table_name: str) -> str:
        with self.engine.connect() as conn:
            result = conn.execute(text(f"DESCRIBE {table_name}"))
            schema = [f"{row[0]} {row[1]}" for row in result]
        return "\n".join(schema)

class RunQueryTool(SQLTool):
    """Tool for running a SQL query"""
    @property
    def name(self) -> str:
        return "run_query"
        
    @property
    def description(self) -> str:
        return "Run a SQL query. Usage: run_query(query)"
        
    def run(self, query: str) -> str:
        try:
            # Check for unsafe operations
            query_lower = query.lower()
            unsafe_operations = ["insert", "update", "delete", "drop", "alter", "truncate", "create"]
            for op in unsafe_operations:
                if op in query_lower and re.search(r'\b' + op + r'\b', query_lower):
                    return f"Error: Unsafe operation detected: {op}. Please use only SELECT statements."
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                rows = [str(row) for row in result]
            return "\n".join(rows)
        except Exception as e:
            return f"Error: {str(e)}"

class QueryWriter:
    def __init__(self, engine: Engine):
        self.instructions = '''
            You are an agent designed to interact with a SQL database.
            Given an input question, create a syntactically correct sql query to answer the question.
            Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
            You can order the results by a relevant column to return the most interesting examples in the database.
            Never query for all the columns from a specific table, only ask for the relevant columns given the question.
            You have access to tools for interacting with the database.
            Only use the below tools. Only use the information returned by the below tools to construct your final answer.
            You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

            DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

            To start you should ALWAYS look at the tables in the database to see what you can query.
            Do NOT skip this step.
            Then you should query the schema of the most relevant tables.

            You do not need to run the query, just provide the SQL query that would answer the question.
            Your job is done once you provide the SQL query.
            You must return the query in the following format:

            ```<query>```
        '''
        self.engine = engine
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize tools
        self.tools = {
            tool.name: tool for tool in [
                ListTablesTool(self.engine),
                GetTableSchemaTool(self.engine),
                RunQueryTool(self.engine)
            ]
        }
        
    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for the prompt"""
        return "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools.values()])
        
    def _get_tool_names(self) -> str:
        """Get a comma-separated list of tool names"""
        return ", ".join(self.tools.keys())
    
    def _execute_tool(self, tool_name: str, tool_input: str = "") -> str:
        """Execute a tool with the given input"""
        if tool_name not in self.tools:
            return f"Error: Tool {tool_name} not found. Available tools: {list(self.tools.keys())}"
            
        tool = self.tools[tool_name]
        
        if tool_input:
            return tool.run(tool_input)
        else:
            return tool.run()
    
    def _parse_action(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse the action and input from the model's response"""
        # Match patterns like Action: tool_name(arguments) or Action: tool_name
        action_match = re.search(r"Action:\s*(\w+)(?:\(([^)]*)\))?", text)
        if not action_match:
            return None, None
            
        tool_name = action_match.group(1)
        tool_input = action_match.group(2) if action_match.group(2) else ""
        
        # Clean up tool input (remove quotes if present)
        tool_input = tool_input.strip('"\'')
        
        return tool_name, tool_input
    
    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Extract the final answer from the text"""
        # Look for "Final Answer:" followed by a SQL query in triple backticks
        final_answer_match = re.search(r"Final Answer:\s*```(?:sql)?(.*?)```", text, re.DOTALL)
        if final_answer_match:
            return final_answer_match.group(1).strip()
        return None

    def _get_response(self, input: List[Dict[str, str]], model_name: str) -> str:
        """
        Get a response from the OpenAI Response API.
        """
        response = self.client.responses.create(
            model=model_name,
            input=input,
            temperature=0
        )
        return response.output_text

    def _get_developer_prompt(self, chat_history: str) -> str:
        """
        Generate the developer prompt for the agent.
        """
        return f"""
            {self.instructions}

            You have access to the following tools:
            
            {self._format_tool_descriptions()}

            Please use the following format:

            ```
            Thought: I need to think about the question and decide if I need to use a tool.
            Action: tool_name(arguments)
            Observation: the result of the action. Will be provided by the tool.
            ```

            ... (repeat Thought/Action/Observation as many times as needed)

            When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

            ```
            Thought: I now know the SQL query to answer the question.
            Final Answer: ```<SQL query>```
            ```

            Do not provide final answer unit you are sure you have the correct SQL query
            and you do not need to use any more tools.

            Begin!

            Previous conversation history:
            {chat_history}
        """

    def generate_query(self, question: str, verbose: bool = False) -> dict:        
        chat_history = []
        for i in range(MAX_ITERATIONS):
            input = [
                {"role": "developer", "content": self._get_developer_prompt(chat_history)},
                {"role": "user", "content": question}
            ]
            response_text = self._get_response(input=input, model_name='gpt-4o-mini')

            chat_history.append({"role": "developer", "content": self._get_developer_prompt(chat_history)})
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": response_text})
            
            if verbose:
                print(f"\n--- Iteration {i+1} ---")
                print("\n--- Chat History ---")
                for entry in chat_history:
                    print(f"{entry['role']}: {entry['content']}")
                print("\n--- LLM Response ---")
                print(f"Response {i+1}: {response_text}")
            
            # Check if the response contains a final answer
            final_answer = self._extract_final_answer(response_text)
            if final_answer:
                # Return when we have a final answer
                return f"```\n{final_answer}\n```"
                
            # Parse the action and input
            tool_name, tool_input = self._parse_action(response_text)
            if not tool_name:
                # If no action found, prompt the model to use a tool correctly
                observation = "I couldn't determine which tool to use. Please use the format 'Action: tool_name(arguments)'."
            else:
                # Execute the tool
                observation = self._execute_tool(tool_name, tool_input)
            
            if verbose:
                print(f"Tool Action: {tool_name}({tool_input})\nObservation: {observation}")



            # Add the observation to the chat history
            chat_history.append({"role": "user", "content": f"Observation: {observation}"})
        
        # If we exceed max iterations without a final answer
        return "Failed to generate a SQL query within the maximum number of iterations. Please try rephrasing your question."

    @staticmethod
    def response_parser(response: str) -> str:
        '''
        Helper fx to parse the response to get the query.
        '''
        return response.replace('```', '').strip()

    def run_query(self, query: str) -> str:
        '''
        Run the query and return the results.
        '''
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            rows = [str(row) for row in result]
        return "\n".join(rows)

if __name__ == "__main__":
    from sample_db.bike_store import BikeStoreDb
    db = BikeStoreDb()
    engine = db.get_engine()
    query_writer = QueryWriter(engine)
    
    # Example usage
    question = "What are the top 5 most expensive bikes?"
    response = query_writer.generate_query(question, verbose=True)
    print("\n--- Generated SQL Query ---\n", query_writer.response_parser(response))
    
    # Run the query
    results = query_writer.run_query(query_writer.response_parser(response).replace('```', ''))
    print("Query Results:", results)