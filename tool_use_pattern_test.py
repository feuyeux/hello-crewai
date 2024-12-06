import os
from crewai import Agent, Task, Crew, LLM
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import SerperDevTool

ollama_llm = LLM(
    model="ollama/llama3.2:1b",
    base_url="http://localhost:11434"
)

search_tool = SerperDevTool()
researcher = Agent(
    role='你是一个天气学家',
    goal='提供今天的天气信息',
    backstory='你入职在气象局，你有专业的分析能力，来分析在特定天气下的穿衣建议',
    #tools=[DirectoryReadTool],
    llm=ollama_llm,
    #是否输出日志信息
    verbose=True,
    #是否与其他的Agent交互
    allow_delegation=False,
    tools=[search_tool]
)
writer = Agent(
    role='你是一个专业的作家',
    goal='在天气预报的基础下，撰写穿衣指南',
    backstory='你很擅长写文章',
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm,
)

#该框架的特点：将角色与任务分开
task1 = Task(
    description='调查今天的天气情况',
    expected_output='一份天气总结',
    agent=researcher
)

task2 = Task(
    description='写一篇穿衣指南',
    expected_output='包括上衣，裤子，鞋',
    agent=writer,
    #output_file='blog-posts/new_post.md' # The final blog post will be saved here
)

# Assemble a crew with planning enabled
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=True
)

result = crew.kickoff()

print("#################")
print(result)