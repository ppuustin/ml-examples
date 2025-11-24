import os, time

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

#from langchain_openai import AzureChatOpenAI
#from langchain_core.messages import HumanMessage

# ------------------------------------

def start_agents(agents, tasks):
    crew = Crew(agents=agents, tasks=tasks, verbose=True)
    result = crew.kickoff()
    print('-------------------------')
    print(result)

def get_agent(role, goal, backstory, max_iter, max_execution_time, description, expected_output, search_tools=None):
    agent = Agent(role=role, goal=goal, backstory=backstory, verbose=True, allow_delegation=False, 
                  max_iter=max_iter, max_execution_time=max_execution_time, tools=search_tools)

    task = Task(description=description, expected_output=expected_output, agent=agent)  
    return agent, task

# ------------------------------------

def get_researcher_defs(identify_what, clarify_what, role_in):
    goal = 'Identify reliable sources with {0} {1}.'.format(identify_what, clarify_what)
    backstory = '''You are a key member of the analytics team at a {0}.
                   With a sharp eye for relevant data, your task is to sift through various 
                   sources to find credible articles containing {1}.
                   Search only recent texts and future ideas.'''.format(role_in, identify_what)
    description = '''Utilize trusted sources to compile a shortlist of articles that are likely to contain information about 
                   {0} {1}. Only shortlist the articles about {2}.
                   Only use the Title, link and snippets from the search results to make the decision. 
                   '''.format(identify_what, clarify_what, identify_what)

    expected_output = 'Curated list of articles with summaries highlighting their relevance.'
    return goal, backstory, description, expected_output

def get_writer_defs(shortlist_of):
    goal = 'Synthesize the shortlisted articles into a concise report'
    backstory = '''As a Report Writer, you're skilled at condensing complex information into clear, actionable insights.
                   Working closely with the Data Analyst, you transform the curated list of articles into an easy-to-understand report 
                   that highlights the links.'''
    description = '''Develop a report based on the shortlisted articles.
                     The report should succinctly present the article names, links and their relevance. 
                     Extract also the {0}.'''.format(shortlist_of)

    expected_output = 'An executive summary report detailing the articles list.'
    return goal, backstory, description, expected_output

def run_agents():
    identify_what = 'python examples for text CNN using Tensorflow'
    clarify_what = 'for keyword identification in text.'
    shortlist_of = 'text CNN methodology types'
    role_in = 'research team'

    researcher_goal, researcher_backstory, T1_DESC, T1_OUTPUT = get_researcher_defs(identify_what, clarify_what, role_in)
    writer_goal, writer_backstory, T2_DESC, T2_OUTPUT = get_writer_defs(shortlist_of)

    props = {}
    props['max_iter'] = 5
    props['max_execution_time'] = 1000

    researcher = props.copy()
    researcher['role'] = 'Researcher'
    researcher['goal'] = researcher_goal
    researcher['backstory'] = researcher_backstory
    researcher['description'] = T1_DESC
    researcher['expected_output'] = T1_OUTPUT
    researcher['search_tools'] = [SerperDevTool()]
     
    writer = props.copy()
    writer['role'] = 'Report Writer'
    writer['goal'] = writer_goal
    writer['backstory'] = writer_backstory
    writer['description'] = T2_DESC
    writer['expected_output'] = T2_OUTPUT
    writer['search_tools'] = None   

    agent1, task1 = get_agent(**researcher)
    agent2, task2 = get_agent(**writer)
    
    start_agents([agent1, agent2], [task1, task2])

if __name__=="__main__":
    start_time = time.time()

    os.environ['SERPER_API_KEY'] =  'xx' # https://serper.dev/
    #os.environ['OPENAI_API_KEY'] = 'xx'
    #os.environ['OPENAI_API_BASE'] = 'xx'
    
    # or use .env
    os.environ['AZURE_API_KEY'] = 'xx'
    os.environ['AZURE_API_BASE'] = 'xx'
    os.environ['AZURE_API_VERSION'] = 'xx'
    os.environ['OPENAI_MODEL_NAME'] = 'azure/gpt-xxx'

    run_agents()
    print('DONE in:', time.time() - start_time, 's')

'''

'''