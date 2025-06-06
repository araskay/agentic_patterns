# Agentic (and non-agentic) patterns
Not all LLM workflows are agentic. Genrally speaking, a workflow is a series of steps to achieve a task. For example, an expense approval workflow may be based on simple rules: “if it is a food expense and less than $50, automatically approve. If more than $50, send to payroll for review”.

Some workflows use LLMs and are often referred to as **AI workflows**. AI workflows can be agentic or non-agentic. In **non-agentic workflows**, the LLM is prompted with an input and generates an output. For example, a text summarization workflow gets a piece of text and returns a shorter summary.

**Agentic workflows** on the other hand often have more autonomy and consist of a series of steps that are dynamically executed by the agent. This requires a reasoning capacity. AI agents often have access to tools that enable them to interact with the world to gather information and accomplish tasks. These agents also have a memory component to enable remembering the context in which previous steps took place to identify the next steps.

This repository includes sample implementations of common agentic and non-agentic patterns. To make it as general as possible, and independent form any specific platform such as Langchain, these patterns are implemented from scratch.

# Non-agentic patterns
## 1. Prompt chaining

![Prompt Chaining Pattern](agentic%20patterns%20-%20prompt%20chaining.png)

The output of one LLM sequentially feeds into the input of the next.

[`prompt_chaining_essay_writer.py`](https://github.com/araskay/agentic_patterns/blob/main/non-agentic_workflows/prompt_chaining_essay_writer.py) contains an example where prompt chaining is used to compose as essay using the following workflow:
1. create an outline
2. expand the outline by adding content to each section
3. write an essay based on the expanded outline

## 2. Routing

![Routing Pattern](agentic%20patterns%20-%20Routing.png)

A routing LLM examines user prompt and sends it to the most appropriate LLM to process.

[`routing_question_answering.py`](https://github.com/araskay/agentic_patterns/blob/main/non-agentic_workflows/routing_question_answering.py) contains an example where routing is used to send user's query to the best LLM. For general questions, a cheaper and faster model, namely `gpt-4o-mini` is used. For coding questions the `gpt-4.1`, which is more suitable for code generation but slower and more expensive, is used.
workflow:
1. determine whether the question is a general question or a coding question.
    Route the question to the appropriate model.
2. if the question is a general question, route it to the general model.
    The general model uses openai's gpt-4o-mini. This is a less expensive model
    and is capable of handling general questions.
3. if the question is a coding question, route it to the code model.
    The code model uses openai's gpt-4.1. This is a more expensive model
    more capable of handling coding questions.

## 3. Parallelization

![Parallelization Pattern](agentic%20patterns%20-%20parallelization.png)

A task is broken down into multiple *independent* subtasks. Each subtask is processed in parallel using a LLM. Subtask results are aggregated to generate the final response.

[`parallelization_development_planner.py`](https://github.com/araskay/agentic_patterns/blob/main/non-agentic_workflows/parallelization_development_planner.py) contains an example where parallelization is used to create a development plan. First, multiple (sub)plans are created by instructing LLMs to follow various personas (e.g., data scientist, software engineer, product manager, UX designer). These subplans are subsequently aggregated by an LLM to create the final plan.

# Agentic patterns
## 1. Reflection

![Reflection Pattern](agentic%20patterns%20-%20reflection.png)

An agent examines its own output and imporves the response based on the critique iteratively.

[`reflection_coding.py`](https://github.com/araskay/agentic_patterns/blob/main/agentic_workflows/reflection_coding.py) contains an example where reflection is used to generate high quality code. The agent prompts a LLM to generate code based on the user query (Genration). The output is subsequently examined in another LLM prompt for correctness and efficiency, where the LLM is instructed to return 'CORRECT' if the code is good, or provide specific feedback for improvement (Reflection). The process of generation-reflection is repeated until "correct" code is generated or a maximum number of iterations reached.

## 2. Tool use

![Tool Use Pattern](agentic%20patterns%20-%20tool%20use.png)

An LLM uses tools (functions, APIs, etc.) to interact with the outside world.

[`tool_use_query_writer.py`](https://github.com/araskay/agentic_patterns/blob/main/agentic_workflows/tool_use_query_writer.py) includes an example where tool use is used by the agent to write SQL queries from natural language.
The agent has access to a list of tools and their descriptions.
These tools enable the agent to interact with the database.
The agent will use these tools to answer the user's question
by going through an iterative process of reasoning and action (the "ReAct" pattern)
as follows:
1. Datermine if I need to use a tool to answer the question.
2. If I need to use a tool, I will use the tool to get the information I need.
    Add the tool's output to the chat history. Go back to step 1 for the next iteration.
3. If I do not need to use a tool, I will provide the final answer based on the information I have.

## 3. Orchestrator-Worker
![Orchestrator-Worker Pattern](agentic%20patterns%20-%20orchestrator-worker.png)

An "orchestrator" or "planner" LLM breaks down a complex task into a *dynamic* list of subtasks. Each subtask is deligated to an agent to complete. A "synthesizer" LLM collects the results from workers and synthesizes the final output.

[`orchestrator-worker_research_agent.py`](https://github.com/araskay/agentic_patterns/blob/main/agentic_workflows/orchestrator-worker_research_agent.py) includes an example where the orchestrator-worker pattern is used to conduct research on a given topic, using the following workflow:
1. User provides a research topic
2. Orchestrator creates a research plan including a list of subtasks
   and deligates subtasks to workers
3. Workers execute subtasks in parallel (when dependencies allow)
4. Synthesizer collects and combines all results into a final output

The main difference between this pattern and Parallelizatoin is that the agent has autonomy in creating the list of subtasks.

## 4. Multi-agent
Multiple agents collaborate to achieve a goal. It is common for the agents to assume a distinct role or persona (project manger, UX designer, coder, tester, etc.) Agents generally operate in an autonomous manner.

There are two common multi-agent patterns:

1. Coordinator / manager pattern
![Multi-agent Coordinator Pattern](agentic%20patterns%20-%20Multi-agent%20coordinator%20approach.png)

2. Swarm pattern
![Multi-agent Swarm Pattern](agentic%20patterns%20-%20Multi-agent%20swarm.png)

# Running the code
## Environment Variables
The following environment variables need to be set before running the code:

- `OPENAI_API_KEY`: Your OpenAI API key. You can get one from [OpenAI's website](https://platform.openai.com/api-keys)
- `EXA_API_KEY`: Your Exa API key. Required for the research agent to search external information. You can get one from [Exa's website](https://exa.ai)

You can set these environment variables by either:
1. Creating a `.env` file in the root directory with the above variables:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   EXA_API_KEY=your-exa-api-key-here
   ```
2. Exporting them in your shell:
   ```bash
   export OPENAI_API_KEY='your-openai-api-key-here'
   export EXA_API_KEY='your-exa-api-key-here'
   ```

