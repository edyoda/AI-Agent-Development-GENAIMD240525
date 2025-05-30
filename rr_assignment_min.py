import asyncio
import os
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.base import TaskResult
from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import List, Optional
import httpx
from pydantic import BaseModel

app = FastAPI()

class TocItem(BaseModel):
    topic: str
    sub_topics: List[str]

# Add this Pydantic model to define the request body
class AssignmentRequest(BaseModel):
    course: str
    course_toc: List[TocItem] 
    topic: str
    sub_topics: List[str]
    assignment_meta_id: str

async def generate_assignments(course, course_toc, topic, sub_topics):
    # Create model clients
    oai_model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        temperature=0.8,  # Slightly reduced for more consistent output
    )
    
    # Using a more creative temperature for the deepseek model
    ds_model_client = OpenAIChatCompletionClient(
        model="deepseek-chat",
        temperature=0.85,
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com/v1",
        model_info={
            "model_name": "deepseek-chat",
            "api_type": "openai",
            "vendor": "deepseek",
            "json_output": False,
            "vision": False,
            "function_calling": False,
            "function_call": False,
            "streaming": True,
            "cost_per_token": 0,
            "family": "deepseek",
        }
    )

    handson_assignment_agent = AssistantAgent(
            name = "Handson_Assignment_Agent",
            system_message = """
            You are an expert in creating hands-on practical assignments for technical courses.

            Your task is to generate 5 distinct, practical assignments that allow students to apply their knowledge.
            Each assignment should:
            1. Be clearly numbered (Assignment 1, Assignment 2, etc.)
            2. Have a descriptive title related to the topic & it's sub_topics.
            3. Include a scenario or context that mimics real-world application
            4. Provide clear objectives/goals
            5. List detailed step-by-step instructions
            6. Include evaluation criteria or expected outcomes
            7. Suggest resources or references if applicable
            8. Strictly ensure that skills needed to solve the assignments should be part of this topic & sub_topics. You have access to course_toc to understand the learner capability. Don't generate assignments which require skills beyond current topic & sub-topics.
            9. Each assignment should be separated by '---'

            Structure each assignment with proper headings and organization.
            Make your assignments challenging but achievable for students who have completed the related course material.
            Each assignment should take approximately 2-3 hours to complete.

            Format your output clearly with the following sections for EACH assignment & NOT in HTML format:
            ```
            ---

            ## Assignment X: [Title]

            ### Scenario
            [Real-world context]

            ### Objectives
            [Clear goals]

            ### Instructions
            [Step-by-step detailed instructions, where each instruction should be expanded to minimum 50 words that helps learner to do the task]

            ### Evaluation Criteria
            [How success will be measured]

            ### Resources
            [Optional helpful resources and STRICTLY valid links]

            ---
            ```

            Ensure that all 5 assignments cover different aspects of the topic and utilize the sub_topics provided.
            """,
            model_client=ds_model_client,
    )


    # Define termination condition
    max_msg_termination = MaxMessageTermination(max_messages=2)

    team = RoundRobinGroupChat([
        handson_assignment_agent,
        ], termination_condition=max_msg_termination)

    prompt = f"""
    I need to generate 5 hands-on practice assignments for a technical course.

    Here are the inputs:
    - **Assignment Course Name**: {course}
    - **Assignment Course Complete TOC**: {course_toc}
    - **Assignment Topic**: {topic}
    - **Assignment Sub-topics**: {', '.join(sub_topics)}

    Please create 5 distinct, practical assignments that allow students to apply their knowledge from this course.
    Each assignment should focus on a different sub-topic or aspect of the main topic.

    Make sure each assignment is clearly numbered and structured with:
    1. The topic & sub_topics are part of a course_toc, so understanding required to solve the problem should be part of topic & sub_topics for which you are creating assignments & definately not after that.
    2. A descriptive title
    3. A realistic scenario
    4. Clear objectives
    5. Step-by-step instructions
    6. Evaluation criteria

    The final output should be well-formatted markdown that can be parsed to extract each assignment independently.
    """

    final_message = None
    async for message in team.run_stream(task=prompt):
        if isinstance(message, TaskResult):
            print("\nStop Reason:", message.stop_reason)
        else:
            print(f"\n{message.source}: {message.content}")
            if message.source == 'Handson_Assignment_Agent':
                final_message = message.content

    return final_message

async def async_gen_task(course, course_toc, topic, sub_topics, meta_id):



    print(f"Background task started {meta_id}")
    result = await generate_assignments(course, course_toc, topic, sub_topics)
    print(f"Background task finished!")

    try:
        async with httpx.AsyncClient() as client:
                response = await client.post(
                        "http://0.0.0.0:5000/callbackassignments/",
                        json={
                            "status": "success",
                            "course": course,
                            "topic": topic,
                            "sub_topics": sub_topics,
                            "meta_id": meta_id,
                            "result": result
                            },
                    timeout=60.0
                )
                print(f"Webhook response status: {response.status_code}")
    except Exception as e:
        print(f"Failed to send webhook: {str(e)}")

@app.post("/generate_assignments")
async def generate_case_studies(background_tasks: BackgroundTasks, request: AssignmentRequest):

    try:
        background_tasks.add_task(
            async_gen_task,
            course=request.course,
            course_toc=request.course_toc,
            topic=request.topic,
            sub_topics=request.sub_topics,
            meta_id=request.assignment_meta_id
        )

        return {
            "status": "success",
            "result": "Async case study generation started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)

