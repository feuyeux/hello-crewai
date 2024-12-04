import os
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from typing import Any, Dict, List, Optional

# Importing crewAI tools
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)

os.environ["OPENAI_API_KEY"] = "NA"
llm = ChatOpenAI(
    model="ollama/llama3.2",  # Use the correct model name for Ollama
    # Ensure this is the correct base URL for your local Ollama service
    base_url="http://localhost:11434/v1"
)

# Instantiate tools
docs_tool = DirectoryReadTool(directory='/Users/hanl5/coding/feuyeux/blog-posts')
file_tool = FileReadTool()
web_rag_tool = WebsiteSearchTool()
# https://serper.dev/api-key
os.environ["SERPER_API_KEY"] = "NA"
search_tool = SerperDevTool()

# Create agents
researcher = Agent(
    role='Market Research Analyst',
    goal='Provide 2024 market analysis of the AI industry',
    backstory='An expert analyst with a keen eye for market trends.',
    tools=[search_tool, web_rag_tool],
    llm=llm,
    verbose=True
)

writer = Agent(
    role='Content Writer',
    goal='Craft engaging blog posts about the AI industry',
    backstory='A skilled writer with a passion for technology.',
    tools=[docs_tool, file_tool],
    llm=llm,
    verbose=True
)

# Define tasks
research = Task(
    description='Research the latest trends in the AI industry and provide a summary.',
    expected_output='A summary of the top 3 trending developments in the AI industry with a unique perspective on their significance.',
    agent=researcher
)

write = Task(
    description='Write an engaging blog post about the AI industry, based on the research analyst’s summary. Draw inspiration from the latest blog posts in the directory.',
    expected_output='A 4-paragraph blog post formatted in markdown with engaging, informative, and accessible content, avoiding complex jargon.',
    agent=writer,
    output_file='blog-posts/new_post.md' # The final blog post will be saved here
)

# Assemble a crew with planning enabled
crew = Crew(
    agents=[researcher, writer],
    tasks=[research, write],
    verbose=True,
    planning=True, # Enable planning feature
)

# Execute tasks
result = crew.kickoff()