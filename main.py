import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import TavilySearchTool

# Load environment variables from .env file
load_dotenv()

# --- 1. Set Up API Keys and LLM ---

# Set your API keys as environment variables
# Note: CrewAI relies on os.environ for keys.
# load_dotenv() handles this, but we can be explicit.
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Initialize the LLM (e.g., GPT-4o)
# We use a powerful model as agents need strong reasoning.
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Initialize the Tavily Search Tool
# This is the same tool as before, just the CrewAI version.
search_tool = TavilySearchTool()

# --- 2. Define Your Agents ---
# An agent is an autonomous unit with a role, goal, and tools.

# Agent 1: The Financial Analyst
analyst = Agent(
    role="Senior Financial Analyst",
    goal="Gather, analyze, and synthesize the latest financial news, "
         "stock performance data, and market sentiment for a given company.",
    backstory=(
        "You are an expert financial analyst with 20 years of experience "
        "at a top-tier investment bank. You are known for your "
        "meticulous research and ability to spot trends that others miss. "
        "You provide clear, data-driven insights."
    ),
    tools=[search_tool],  # This agent can use the search tool
    llm=llm,
    verbose=True,
    allow_delegation=False,  # This agent cannot delegate work
)

# Agent 2: The Report Writer
writer = Agent(
    role="Financial Report Writer",
    goal="Write a comprehensive, professional, and easy-to-understand "
         "financial report based on the analysis provided by the "
         "Senior Financial Analyst.",
    backstory=(
        "You are a skilled financial writer, a former journalist from "
        "The Wall Street Journal. You specialize in translating complex "
        "financial data into compelling narratives that are perfect for "
        "executives and investors. You do not do research yourself."
    ),
    tools=[],  # This agent does not have tools
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# --- 3. Define the Tasks ---
# A task is a specific unit of work to be done by an agent.

# Task 1: Research the company
research_task = Task(
    description=(
        "Conduct a comprehensive analysis of the company: {topic}. "
        "Gather the latest news, press releases, quarterly earnings reports, "
        "and overall market sentiment. Find key financial metrics, "
        "recent developments, and future outlook."
    ),
    expected_output=(
        "A detailed, bullet-point summary of all relevant findings, "
        "including quantitative data, key news headlines, and a summary "
        "of the market's view on the company. "
        "This will be the raw material for the writer."
    ),
    agent=analyst,  # Assign the task to the analyst
)

# Task 2: Write the final report
write_report_task = Task(
    description=(
        "Write a polished, final report on {topic} based on the "
        "research findings. The report must be structured, clear, "
        "and insightful, suitable for a board of directors."
    ),
    expected_output=(
        "A 4-paragraph comprehensive report on {topic}, "
        "including an introduction, a summary of key findings, "
        "an analysis of the current situation, and a concluding outlook."
    ),
    agent=writer,  # Assign the task to the writer
    # context=[research_task], # This tells the writer to use the output of the research task
)

# --- 4. Create and Kick Off the Crew ---
# The Crew brings the agents and tasks together.

crew = Crew(
    agents=[analyst, writer],
    tasks=[research_task, write_report_task],
    process=Process.sequential,  # Tasks will be executed one after another
    verbose=True  # MODIFIED: Changed from '2' to 'True' to fix the boolean parsing error
)

# Get the topic from the user
print("--- Welcome to the AI Stock Analysis Crew ---")
topic = input("What company would you like to research? (e.g., 'Tesla' or 'NVIDIA')\n> ")

# Kick off the crew's work!
print("\nWorking on it... This may take a few minutes.")
inputs = {"topic": topic}
result = crew.kickoff(inputs=inputs)

print("\n--- Final Report ---")
print(result)

