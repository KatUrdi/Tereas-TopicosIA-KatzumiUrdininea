import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import FileReadTool, CodeDocsSearchTool


load_dotenv()

llm = LLM(model="gpt-4o-mini")
current_dir = os.path.dirname(os.path.abspath(__file__))

def create_agent(role, goal, backstory, llm, allow_code_execution=False):
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        allow_code_execution=allow_code_execution
    )

analyzer_agent = create_agent(
    role="Python Code Analyzer",
    goal="Analyze documents and provide relevant code insights or translate them into code.",
    backstory="""You are an expert Python developer tasked with reading and analyzing a scientific paper.
    The scientific paper is in PDF format, and you are responsible for analyzing or translating the described code.
    You will also reference the FastAPI and SciPy documentation to implement the code in Python.""",
    llm=llm
)

coder_agent = create_agent(
    role="Senior Python Developer",
    goal="Generate code based on the analysis and document it in a Python file.",
    backstory="""You are a senior Python developer experienced in translating analysis into clean, executable code.
    You will generate Python code based on the analysis provided, using FastAPI and SciPy documentation. 
    The code must be clean, well-documented, and executable.""",
    llm=llm,
    allow_code_execution=True
)

def create_task(description, expected_output, agent, tools=None, output_file=None):
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        tools=tools or [],
        output_file=output_file
    )

read_pdf_task = create_task(
    description="Read the provided PDF document and analyze the code or translate it into Python code.",
    expected_output="A detailed analysis of the code in the PDF document.",
    agent=analyzer_agent,
    tools=[FileReadTool(file_path=os.path.join(current_dir, "Lect-7-DM.pdf"))]
)

fetch_fastapi_docs_task = create_task(
    description="Fetch and analyze the FastAPI documentation.",
    expected_output="A summary of relevant FastAPI documentation points for the code.",
    agent=analyzer_agent,
    tools=[CodeDocsSearchTool(query="https://fastapi.tiangolo.com/")]
)

fetch_scipy_docs_task = create_task(
    description="Fetch and analyze the SciPy documentation.",
    expected_output="A summary of relevant SciPy documentation points for the code.",
    agent=analyzer_agent,
    tools=[CodeDocsSearchTool(query="https://docs.scipy.org/doc/scipy/tutorial/index.html#user-guide")]
)

generate_code_task = create_task(
    description="Generate an executable Python script based on the PDF document, FastAPI, and SciPy documentation analysis.",
    expected_output="A valid Python `.py` file implementing the functionality described in the PDF, with clear comments.",
    agent=coder_agent,
    output_file="replicated_gen.py"
)

dev_crew = Crew(
    agents=[analyzer_agent, coder_agent],
    tasks=[read_pdf_task, fetch_fastapi_docs_task, fetch_scipy_docs_task, generate_code_task],
    verbose=True
)

result = dev_crew.kickoff()

print(result)
