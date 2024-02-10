# Defining the necessary libraries
import os
from langchain.agents import tool
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser
)
from langchain.tools.render import render_text_description
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models.huggingface import ChatHuggingFace

# Connecting to huggingface hub
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<INSERT YOUR TOKEN HERE>"

# LLM instantiation
llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 1024,
        "temperature": 0.1
    }
)

# Instantiate the chat model
chat_model = ChatHuggingFace(llm=llm)


# Define the tools
@tool
def salary_hike_calculator(points):
    '''use this tool when you need to calculate the salary hike
            input: no.of points (float)
            output: hike in rupees (integer)
            hike calculation = no.of points X 1000
    '''
    return points*1000


@tool
def rating_calculator(points):
    '''use this tool when you need to calculate the rating
        input: no.of points (float)
        output: rating (string)
        if no.of points>3, rating=good
        if no.of points<3, rating=bad
        if no.of points>5 or no.of points<0, rating= none
    '''
    if points > 3:
        return "good"
    elif points < 3 and points > 0:
        return "bad"
    else:
        return "none"


tools = [salary_hike_calculator, rating_calculator]

# Construct or import the prompt from langchain hub
prompt = hub.pull("hwchase17/react-json")

prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# Build an Agent
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(
            x["intermediate_steps"]
            ),
    }
    | prompt
    | chat_model
    | ReActJsonSingleInputOutputParser()
)

# instantiate AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Invoke the AgentExecutor
print(list(agent_executor.stream(
    {"input": '''likhith has received 2 points.
    what is the expected salary hike and rating?'''}
  )))
