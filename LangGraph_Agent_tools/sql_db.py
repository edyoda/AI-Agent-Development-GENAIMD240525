import os
from langchain.agents import AgentType
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from model_inference import llm

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent

# Setup the database
# Make the required changes to this wherever required.
# https://python.langchain.com/docs/integrations/tools/sql_database/



# Make sure to have the SQLite database file 'chinook.db' in the 'database' directory
db_path = os.path.join(os.path.dirname(__file__), "database", "chinook.db")

db = SQLDatabase.from_uri("sqlite:////" + db_path)

agent = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# agent.run("List all albums by the artist 'AC/DC'")