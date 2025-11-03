import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import TavilySearchTool

# --- 1. LANGSMITH INSTRUMENTATION SETUP (NEW) ---
# Import the required OpenTelemetry and LangSmith packages
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
# MODIFIED: Removed SimpleSpanProcessor, as LangSmith's processor handles this
# MODIFIED: Updated the import path for the new LangSmith processor
from langsmith.integrations.otel import OtelSpanProcessor 
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor

# Set up the TracerProvider
# Get the current TracerProvider, or create one if it doesn't exist
provider = trace.get_tracer_provider()
if not isinstance(provider, TracerProvider):
    provider = TracerProvider()
    trace.set_tracer_provider(provider)

# MODIFIED: Add the LangSmith OtelSpanProcessor
provider.add_span_processor(OtelSpanProcessor())

# Instrument CrewAI and OpenAI
CrewAIInstrumentor().instrument()
OpenAIInstrumentor().instrument()
# --- END OF INSTRUMENTATION SETUP ---


# --- 2. Load Environment Variables ---
# This will now load your .env file, including the new LangSmith keys
load_dotenv()

# Set your API keys (this part is the same)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# --- 3. Set Up LLM and Tools (Same as before) ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)
search_tool = TavilySearchTool()

# --- 4. Define Your Agents (Same as before) ---
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
    tools=[search_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

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
    tools=[],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# --- 5. Define the Tasks (Same as before) ---
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
    agent=analyst,
)

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
    agent=writer,
)

# --- 6. Create and Kick Off the Crew (Same as before) ---
crew = Crew(
    agents=[analyst, writer],
    tasks=[research_task, write_report_task],
    process=Process.sequential,
    verbose=True
)

# Get the topic from the user
print("--- Welcome to the AI Stock Analysis Crew (with LangSmith) ---")
topic = input("What company would you like to research? (e.g., 'Tesla' or 'NVIDIA')\n> ")

# Kick off the crew's work!
print("\nWorking on it... This may take a few minutes.")
inputs = {"topic": topic}
result = crew.kickoff(inputs=inputs)

print("\n--- Final Report ---")
print(result)

# --- 7. Link to LangSmith Trace (NEW) ---
print("\n---")
print("Trace complete. Check your LangSmith project to see the full run:")
# Try to get the project name from the environment, fall back to a default if not set
project_name = os.getenv("LANGCHAIN_PROJECT", "default")
print(f"(Look for the project named '{project_name}')")
print("---")

