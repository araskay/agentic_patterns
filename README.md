# Agentic (and non-agentic) patterns
Sample implementations of common agentic and non-agentic patterns from scratch

# Non-agentic patterns
## 1. Prompt chaining
The output of one LLM sequentially feeds into the input of the next.

`prompt_chaining_essay_writer.py` contains an example where prompt chaining is used to compose as essay using the following workflow:
1. create an outline
2. expand the outline by adding content to each section
3. write an essay based on the expanded outline

## 2. Routing
A routing LLM examines user prompt and sends it to the most appropriate LLM to process.

`routing_question_answering.py` contains an example where routing is used to send user's query to the best LLM. For general questions, a cheaper and faster model, namely `gpt-4o-mini` is used. For coding questions the `gpt-4.1`, which is more suitable for code generation but slower and more expensive, is used.
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
A task is broken down into multiple *independent* subtasks. Each subtask is processed in parallel using a LLM. Subtask results are aggregated to generate the final response.

`parallelization_development_planner.py` contains an example where parallelization is used to create a development plan. First, multiple (sub)plans are created by instructing LLMs to follow various personas (e.g., data scientist, software engineer, product manager, UX designer). These subplans are subsequently aggregated by an LLM to create the final plan.

# Agentic patterns
## 1. Reflection

## 2. Tool use

## 3. Planning

## 4. Multi-agent
