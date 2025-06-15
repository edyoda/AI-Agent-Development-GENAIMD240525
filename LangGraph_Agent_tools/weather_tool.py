from langchain.agents import Tool
from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.agents import AgentType
from langchain.agents import initialize_agent

from model_inference import llm



# https://python.langchain.com/docs/integrations/tools/openweathermap/


# Initialize API tool
# weather = OpenWeatherMapAPIWrapper(openweathermap_api_key="your_api_key_here")

def weather(location: str):
    """
    Fetches the weather for a given location.
    
    Args:
        location (str): The location for which to fetch the weather.
        
    Returns:
        str: Weather information for the specified location.
    """
    return f"Weather in {location}: Sunny, 25°C with a light breeze."

weather_tool = Tool(
    name="WeatherAPI",
    func=weather,
    description="Useful for fetching weather data",
)

agent = initialize_agent(
    tools=[weather_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# response = agent.run("What's the weather in Berlin tomorrow?")

# print(response)