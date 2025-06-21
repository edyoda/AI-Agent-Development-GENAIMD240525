from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import random
from datetime import datetime
import json
import openai
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent as ChatAssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen import UserProxyAgent, config_list_from_dotenv
import re
import asyncio
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import aiofiles
import aiohttp
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType
from playwright.async_api import async_playwright
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from autogen_ext.models.openai import OpenAIChatCompletionClient
from contextlib import asynccontextmanager
from autogen_agentchat.agents import AssistantAgent

# Load environment variables
load_dotenv()

# Global variables for course recommendation system
rag_memory = None
course_recommendation_agent = None

oai_model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    temperature=0.85,
    api_key=os.getenv("OPENAI_API_KEY"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    # Initialize course recommendation system in background
    asyncio.create_task(initialize_course_recommendation_system())
    yield
    # Shutdown
    if rag_memory:
        try:
            await rag_memory.close()
        except Exception as e:
            print(f"Error closing memory: {e}")

app = FastAPI(
    title="Skill Quiz API with Course Recommendations", 
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Valid domains
VALID_DOMAINS = [
    "GenAI", "Cloud", "DevOps", "Drone", "Robotics", 
    "Data Engineering", "Cybersecurity", "Space Technology", 
    "Analytics", "Business", "Management"
]

# Course Recommendation System Components
class SimpleDocumentIndexer:
    """Basic document indexer for AutoGen Memory."""

    def __init__(self, memory: Memory, chunk_size: int = 1500) -> None:
        self.memory = memory
        self.chunk_size = chunk_size

    async def _fetch_content(self, source: str) -> str:
        """Fetch fully rendered content from URL using Playwright."""
        if source.startswith(("http://", "https://")):
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(source, wait_until="networkidle")
                content = await page.content()
                await browser.close()
                return content
        else:
           async with aiofiles.open(source, "r", encoding="utf-8") as f:
               return await f.read()

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags and normalize whitespace."""
        text = re.sub(r"<[^>]*>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _split_text(self, text: str) -> List[str]:
        """Split text into fixed-size chunks."""
        chunks: list[str] = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i : i + self.chunk_size]
            chunks.append(chunk.strip())
        return chunks

    async def index_documents(self, sources: List[str]) -> int:
        """Index documents into memory."""
        total_chunks = 0

        for source in sources:
            try:
                content = await self._fetch_content(source)

                if "<" in content and ">" in content:
                    content = self._strip_html(content)

                chunks = self._split_text(content)

                for i, chunk in enumerate(chunks):
                    await self.memory.add(
                        MemoryContent(
                            content=chunk, 
                            mime_type=MemoryMimeType.TEXT, 
                            metadata={"source": source, "chunk_index": i}
                        )
                    )

                total_chunks += len(chunks)
                print(f"Indexed {len(chunks)} chunks from {source}")

            except Exception as e:
                print(f"Error indexing {source}: {str(e)}")

        return total_chunks

def extract_urls_from_domain(domain):
    """Extracts all URLs from a given domain."""
    urls = set()
    try:
        response = requests.get(domain)
        soup = BeautifulSoup(response.content, 'html.parser')

        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(domain, href)
            if full_url.startswith(domain):
                urls.add(full_url)

    except requests.exceptions.RequestException as e:
        print(f"Error while fetching {domain}: {e}")

    return list(urls)

# Global variables for course recommendation system
rag_memory = None
course_recommendation_agent = None

async def initialize_course_recommendation_system():
    """Initialize the course recommendation system with EdYoda data."""
    global rag_memory, course_recommendation_agent
    
    try:
        # Initialize vector memory with proper persistence
        persistence_path = os.path.join(str(Path.home()), ".chromadb_autogen")
        print(f"Using persistence path: {persistence_path}")

        rag_memory = ChromaDBVectorMemory(
            config=PersistentChromaDBVectorMemoryConfig(
                collection_name="edyoda_courses",
                persistence_path=persistence_path,
                k=5,  # Return top 5 results for better recommendations
                score_threshold=0.3,  # Lower threshold for more inclusive results
            )
        )

        # Check if documents are already indexed
        try:
            import chromadb
            client = chromadb.PersistentClient(path=persistence_path)
            
            try:
                collection = client.get_collection("edyoda_courses")
                count = collection.count()
                print(f"Found existing collection with {count} documents")
                
                if count == 0:
                    await index_edyoda_courses()
                    
            except Exception:
                print("Collection doesn't exist, indexing courses...")
                await index_edyoda_courses()
                
        except Exception as e:
            print(f"Error checking collection: {e}")
            await index_edyoda_courses()

        # Create course recommendation assistant
        course_recommendation_agent = AssistantAgent(
            name="course_recommendation_agent", 
            model_client=oai_model_client,
            memory=[rag_memory],
            system_message="""You are a career guidance expert specializing in recommending EdYoda courses based on skill assessment feedback. 

Your role is to:
1. Analyze the provided feedback and identified skill gaps
2. Recommend specific EdYoda courses that address these gaps
3. Explain how each recommended course will help improve the user's skills
4. Prioritize courses based on the user's current level and career goals
5. Always include the source URLs for recommended courses when available

Guidelines:
- Focus on practical, actionable recommendations
- Consider the user's domain and current skill level
- Recommend 2-4 most relevant courses
- Explain the learning outcomes and career benefits
- Be encouraging and supportive in your tone"""
        )

        print("Course recommendation system initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing course recommendation system: {e}")
        # Set to None so we know it failed but don't crash the app
        rag_memory = None
        course_recommendation_agent = None

async def index_edyoda_courses():
    """Index EdYoda course content."""
    try:
        print("Starting to index EdYoda courses...")
        
        # Extract URLs from EdYoda domain
        all_urls = extract_urls_from_domain("https://www.edyoda.com")
        filtered_urls = [url for url in all_urls if url.rstrip('/').endswith('micro-degree')]
        
        if not filtered_urls:
            print("No course URLs found, using sample URLs")
            # Fallback URLs in case scraping fails
            filtered_urls = [
                "https://www.edyoda.com/course/data-science-micro-degree",
                "https://www.edyoda.com/course/machine-learning-micro-degree",
                "https://www.edyoda.com/course/cloud-computing-micro-degree"
            ]

        print(f"Found {len(filtered_urls)} course URLs to index")
        
        indexer = SimpleDocumentIndexer(memory=rag_memory)
        chunks = await indexer.index_documents(filtered_urls)
        print(f"Successfully indexed {chunks} chunks from {len(filtered_urls)} course pages")
        
    except Exception as e:
        print(f"Error indexing EdYoda courses: {e}")

async def get_course_recommendations(feedback: str, domain: str, skills: List[str], score: int) -> str:
    """Get course recommendations based on evaluation feedback."""
    global course_recommendation_agent, rag_memory
    
    if not course_recommendation_agent:
        return "Course recommendation system is not available. Please contact support."
    
    try:
        skills_text = ", ".join(skills)
        
        # First, retrieve relevant course information from memory
        query_text = f"{domain} {skills_text} course training"
        memory_results = await rag_memory.query(query_text)
        
        # Extract source URLs from memory results
        source_urls = []
        course_content = []
        
        for result in memory_results:
            if hasattr(result, 'metadata') and 'source' in result.metadata:
                source_url = result.metadata['source']
                if source_url not in source_urls:
                    source_urls.append(source_url)
                    course_content.append(f"Course: {source_url}\nContent: {result.content[:500]}...")
        
        course_context = "\n\n".join(course_content)
        
        # Create a comprehensive prompt for course recommendations
        recommendation_prompt = f"""
Based on the following skill assessment results and available EdYoda courses, please recommend relevant courses for career growth:

**Domain:** {domain}
**Skills Assessed:** {skills_text}
**Assessment Score:** {score}/10
**Detailed Feedback:** {feedback}

**Available Course Information:**
{course_context}

**Available Course URLs:**
{chr(10).join(source_urls)}

**Task:** 
Please recommend 2-4 EdYoda courses that would help improve the identified skill gaps and advance the user's career in {domain}. For each recommendation:

1. **Course name and EXACT URL** (use the URLs provided above)
2. How it addresses the specific skill gaps mentioned in the feedback
3. Expected learning outcomes
4. Career benefits and growth opportunities

**IMPORTANT:** Always include the complete course URL from the list above for each recommended course.

Focus on courses that are most relevant to the weaknesses identified in the assessment feedback.
"""

        # Get recommendations using the RAG agent
        response = await course_recommendation_agent.run(task=recommendation_prompt)
        
        # Extract the response content
        if hasattr(response, 'messages') and response.messages:
            recommendation_text = response.messages[-1].content
        else:
            recommendation_text = str(response)
       
        print(recommendation_text)
        return recommendation_text
        
    except Exception as e:
        print(f"Error getting course recommendations: {e}")
        return f"Unable to generate course recommendations at this time. Please try again later. Error: {str(e)}"


# Database initialization
def init_db():
    conn = sqlite3.connect('quiz_app.db')
    cursor = conn.cursor()
    
    # Create questions table (now stores generated questions for reference)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT NOT NULL,
            skills TEXT NOT NULL,
            question TEXT NOT NULL,
            difficulty_level TEXT DEFAULT 'intermediate',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create answers table with course recommendations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER,
            question_text TEXT NOT NULL,
            answer TEXT NOT NULL,
            domain TEXT NOT NULL,
            skills TEXT NOT NULL,
            ai_feedback TEXT,
            ai_score INTEGER,
            course_recommendations TEXT,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (question_id) REFERENCES questions (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Pydantic models
class QuestionRequest(BaseModel):
    domain: str
    skills: List[str]
    difficulty_level: Optional[str] = "intermediate"

class QuestionResponse(BaseModel):
    id: Optional[int] = None
    domain: str
    skills: List[str]
    question: str
    difficulty_level: str

class AnswerRequest(BaseModel):
    question_id: Optional[int] = None
    question_text: str
    answer: str
    domain: str
    skills: List[str]

class AnswerResponse(BaseModel):
    success: bool
    message: str
    ai_feedback: Optional[str] = None
    ai_score: Optional[int] = None
    course_recommendations: Optional[str] = None

async def generate_question_with_autogen(domain: str, skills: List[str], difficulty_level: str = "intermediate") -> str:
    """
    Use AutoGen AssistantAgent to generate a question based on domain and skills.
    """
    if domain not in VALID_DOMAINS:
        raise HTTPException(status_code=400, detail=f"Invalid domain. Valid domains are: {', '.join(VALID_DOMAINS)}")

    question_generator = AssistantAgent(
        name="question_generator",
        model_client=oai_model_client,
        system_message=f"""You are an expert question generator specializing in {domain}.
        Your task is to create high-quality, practical questions that test real-world knowledge and skills.
        
        Guidelines:
        - Create questions that are relevant to current industry practices
        - Focus on practical, scenario-based questions when possible
        - Ensure questions test deep understanding, not just memorization
        - Make questions clear and unambiguous
        - Avoid overly theoretical questions unless specifically relevant
        - Consider different difficulty levels appropriately
        """
    )

    skills_text = ", ".join(skills)
    
    # Create specialized prompts based on domain
    domain_specific_prompts = {
        "GenAI": f"Generate a {difficulty_level}-level question about Generative AI focusing on {skills_text}. Consider topics like model architecture, training, deployment, ethics, or practical applications.",
        
        "Cloud": f"Generate a {difficulty_level}-level question about Cloud Computing focusing on {skills_text}. Consider topics like architecture, services, security, cost optimization, or best practices.",
        
        "DevOps": f"Generate a {difficulty_level}-level question about DevOps focusing on {skills_text}. Consider topics like CI/CD, infrastructure as code, monitoring, containerization, or automation.",
        
        "Drone": f"Generate a {difficulty_level}-level question about Drone Technology focusing on {skills_text}. Consider topics like flight control systems, regulations, applications, hardware, or programming.",
        
        "Robotics": f"Generate a {difficulty_level}-level question about Robotics focusing on {skills_text}. Consider topics like control systems, sensors, actuators, programming, or applications.",
        
        "Data Engineering": f"Generate a {difficulty_level}-level question about Data Engineering focusing on {skills_text}. Consider topics like data pipelines, ETL/ELT, data warehousing, streaming, or data quality.",
        
        "Cybersecurity": f"Generate a {difficulty_level}-level question about Cybersecurity focusing on {skills_text}. Consider topics like threat analysis, security frameworks, incident response, or risk management.",
        
        "Space Technology": f"Generate a {difficulty_level}-level question about Space Technology focusing on {skills_text}. Consider topics like satellite systems, orbital mechanics, space missions, or space communications.",
        
        "Analytics": f"Generate a {difficulty_level}-level question about Analytics focusing on {skills_text}. Consider topics like data analysis, statistical modeling, visualization, or business intelligence.",
        
        "Business": f"Generate a {difficulty_level}-level question about Business focusing on {skills_text}. Consider topics like strategy, operations, finance, marketing, or innovation.",
        
        "Management": f"Generate a {difficulty_level}-level question about Management focusing on {skills_text}. Consider topics like leadership, team management, project management, or organizational behavior."
    }
    
    base_prompt = domain_specific_prompts.get(domain, 
        f"Generate a {difficulty_level}-level question about {domain} focusing on {skills_text}.")
    
    full_prompt = f"""{base_prompt}

Requirements:
- The question should be practical and relevant to current industry needs
- It should test understanding and application, not just recall
- Make it clear and specific
- Ensure it can be answered in 2-3 paragraphs
- Focus on real-world scenarios when possible

Please provide only the question text, nothing else."""

    try:
        # Use the simple run method to get the response
        response = await question_generator.run(task=full_prompt)
        
        # Extract the response content
        if hasattr(response, 'messages') and response.messages:
            question = response.messages[-1].content.strip()
        else:
            question = str(response).strip()
        
        # Clean up the response to ensure we only get the question
        question = question.replace("Question:", "").strip()
        if question.startswith('"') and question.endswith('"'):
            question = question[1:-1]
            
        return question
        
    except Exception as e:
        print(f"Error in AutoGen question generation: {e}")
        return f"Explain the best practices for implementing {skills_text} in {domain} and discuss potential challenges you might face in a real-world scenario."

async def verify_answer_with_autogen(question: str, answer: str, domain: str, skills: List[str]) -> tuple[str, int]:
    """
    Use AutoGen AssistantAgent to evaluate and score the answer.
    """
    # Create assistant agent with proper configuration
    evaluator = AssistantAgent(
        name="evaluator",
        model_client = oai_model_client,
        system_message=f"""You are an expert evaluator specializing in {domain}. 
        Your role is to provide fair, constructive, and detailed feedback on technical answers.
        
        Evaluation Criteria:
        - Technical accuracy and correctness
        - Depth of understanding demonstrated
        - Practical relevance and real-world application
        - Clarity and organization of the response
        - Coverage of key concepts
        
        Always provide constructive feedback that helps the learner improve.""",
    )

    skills_text = ", ".join(skills)
    prompt = f"""
Please evaluate the following answer to a {domain} question focusing on {skills_text}:

QUESTION: {question}

STUDENT ANSWER: {answer}

Please provide your evaluation in this format:

**EVALUATION:**
[Provide 2-3 sentences evaluating the overall quality of the answer]

**STRENGTHS:**
[List what was done well]

**AREAS FOR IMPROVEMENT:**
[List specific areas that could be enhanced]

**RECOMMENDATIONS:**
[Provide specific suggestions for improvement]

**FINAL SCORE: X/10**
[Where X is a number from 1-10 based on: 1-3=Poor, 4-6=Average, 7-8=Good, 9-10=Excellent]

Scoring Guidelines:
- 9-10: Comprehensive, accurate, well-structured with excellent examples
- 7-8: Good understanding, mostly accurate with some good examples
- 5-6: Basic understanding, some accuracy issues or missing key points
- 3-4: Limited understanding, several inaccuracies or very incomplete
- 1-2: Poor understanding, major inaccuracies or completely off-topic
"""

    try:
        # Use the simple run method to get the response
        response = await evaluator.run(task=prompt)
        
        # Extract the response content
        if hasattr(response, 'messages') and response.messages:
            response_text = response.messages[-1].content
        else:
            response_text = str(response)
            
    except Exception as e:
        print(f"Error in AutoGen evaluation: {e}")
        response_text = f"Your answer demonstrates understanding of {domain} concepts related to {skills_text}. Consider providing more specific examples and technical details to strengthen your response. **FINAL SCORE: 6/10**"

    # Extract score from response
    score = 6  # Default fallback
    
    # Look for score patterns
    score_patterns = [
        r'FINAL SCORE:\s*(\d+)/10',
        r'Final Score:\s*(\d+)/10',
        r'Score:\s*(\d+)/10',
        r'(\d+)/10',
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            try:
                potential_score = int(match.group(1))
                if 1 <= potential_score <= 10:
                    score = potential_score
                    break
            except (ValueError, IndexError):
                continue

    return response_text, score

@app.on_event("startup")
async def startup_event():
    init_db()
    # Initialize course recommendation system in background
    asyncio.create_task(initialize_course_recommendation_system())

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the index.html at root path
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)
'''    
@app.get("/")
async def root():
    return {"message": "Skill Quiz API with Course Recommendations is running!"}
'''

@app.get("/domains")
async def get_domains():
    """Get all valid domains"""
    return {"domains": VALID_DOMAINS}

@app.post("/get-question", response_model=QuestionResponse)
async def get_question(request: QuestionRequest):
    """
    Generate a question using AutoGen based on domain and skills
    """
    if not request.skills:
        raise HTTPException(status_code=400, detail="At least one skill must be provided")
    
    if request.domain not in VALID_DOMAINS:
        raise HTTPException(status_code=400, detail=f"Invalid domain. Valid domains are: {', '.join(VALID_DOMAINS)}")
    
    # Generate question using AutoGen
    question_text = await generate_question_with_autogen(
        request.domain, 
        request.skills, 
        request.difficulty_level
    )
    
    # Store the generated question in database for reference
    conn = sqlite3.connect('quiz_app.db')
    cursor = conn.cursor()
    
    skills_json = json.dumps(request.skills)
    cursor.execute(
        """
        INSERT INTO questions (domain, skills, question, difficulty_level) 
        VALUES (?, ?, ?, ?)
        """,
        (request.domain, skills_json, question_text, request.difficulty_level)
    )
    
    question_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return QuestionResponse(
        id=question_id,
        domain=request.domain,
        skills=request.skills,
        question=question_text,
        difficulty_level=request.difficulty_level
    )

@app.post("/submit-answer", response_model=AnswerResponse)
async def submit_answer(request: AnswerRequest):
    """
    Submit an answer to a question and get AI verification with course recommendations
    """
    if not request.answer.strip():
        raise HTTPException(status_code=400, detail="Answer cannot be empty")
    
    # Get AI feedback for the answer
    ai_feedback, ai_score = await verify_answer_with_autogen(
        request.question_text, request.answer, request.domain, request.skills
    )
    
    # Get course recommendations based on the feedback
    course_recommendations = await get_course_recommendations(
        ai_feedback, request.domain, request.skills, ai_score
    )
    
    # Store the answer with AI feedback and course recommendations
    conn = sqlite3.connect('quiz_app.db')
    cursor = conn.cursor()
    
    skills_json = json.dumps(request.skills)
    cursor.execute(
        """
        INSERT INTO answers (question_id, question_text, answer, domain, skills, ai_feedback, ai_score, course_recommendations) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (request.question_id, request.question_text, request.answer, request.domain, skills_json, ai_feedback, ai_score, course_recommendations)
    )
    
    conn.commit()
    conn.close()
    
    return AnswerResponse(
        success=True, 
        message="Answer submitted and evaluated successfully with course recommendations",
        ai_feedback=ai_feedback,
        ai_score=ai_score,
        course_recommendations=course_recommendations
    )

@app.get("/questions")
async def get_all_questions():
    """
    Get all generated questions (for admin purposes)
    """
    conn = sqlite3.connect('quiz_app.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, domain, skills, question, difficulty_level, created_at FROM questions ORDER BY created_at DESC")
    questions = cursor.fetchall()
    
    conn.close()
    
    result = []
    for q in questions:
        try:
            skills_list = json.loads(q[2])
        except:
            skills_list = [q[2]]
        
        result.append({
            "id": q[0],
            "domain": q[1],
            "skills": skills_list,
            "question": q[3],
            "difficulty_level": q[4],
            "created_at": q[5]
        })
    
    return result

@app.get("/answers")
async def get_all_answers():
    """
    Get all answers with course recommendations (for admin purposes)
    """
    conn = sqlite3.connect('quiz_app.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT a.id, a.question_id, a.question_text, a.answer, a.domain, 
               a.skills, a.submitted_at, a.ai_feedback, a.ai_score, a.course_recommendations
        FROM answers a
        ORDER BY a.submitted_at DESC
    """)
    
    answers = cursor.fetchall()
    conn.close()
    
    result = []
    for a in answers:
        try:
            skills_list = json.loads(a[5])
        except:
            skills_list = [a[5]]
        
        result.append({
            "id": a[0],
            "question_id": a[1],
            "question_text": a[2],
            "answer": a[3],
            "domain": a[4],
            "skills": skills_list,
            "submitted_at": a[6],
            "ai_feedback": a[7],
            "ai_score": a[8],
            "course_recommendations": a[9]
        })
    
    return result

@app.get("/analytics")
async def get_analytics():
    """
    Get analytics about questions and answers
    """
    conn = sqlite3.connect('quiz_app.db')
    cursor = conn.cursor()
    
    # Get question count by domain
    cursor.execute("""
        SELECT domain, COUNT(*) as question_count 
        FROM questions 
        GROUP BY domain 
        ORDER BY question_count DESC
    """)
    questions_by_domain = dict(cursor.fetchall())
    
    # Get average scores by domain
    cursor.execute("""
        SELECT domain, AVG(ai_score) as avg_score, COUNT(*) as answer_count
        FROM answers 
        WHERE ai_score IS NOT NULL
        GROUP BY domain 
        ORDER BY avg_score DESC
    """)
    scores_by_domain = {}
    for row in cursor.fetchall():
        scores_by_domain[row[0]] = {
            "average_score": round(row[1], 2),
            "answer_count": row[2]
        }
    
    # Get recent activity
    cursor.execute("""
        SELECT COUNT(*) as total_questions_generated
        FROM questions 
        WHERE created_at >= datetime('now', '-7 days')
    """)
    recent_questions = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(*) as total_answers_submitted
        FROM answers 
        WHERE submitted_at >= datetime('now', '-7 days')
    """)
    recent_answers = cursor.fetchone()[0]
    
    # Get course recommendation usage
    cursor.execute("""
        SELECT COUNT(*) as answers_with_recommendations
        FROM answers 
        WHERE course_recommendations IS NOT NULL AND course_recommendations != ''
    """)
    recommendations_generated = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "questions_by_domain": questions_by_domain,
        "scores_by_domain": scores_by_domain,
        "recent_activity": {
            "questions_generated_this_week": recent_questions,
            "answers_submitted_this_week": recent_answers,
            "course_recommendations_generated": recommendations_generated
        }
    }

@app.get("/course-recommendation-status")
async def get_course_recommendation_status():
    """Check if the course recommendation system is ready"""
    global course_recommendation_agent, rag_memory
    
    return {
        "course_system_ready": course_recommendation_agent is not None,
        "memory_initialized": rag_memory is not None,
        "status": "ready" if course_recommendation_agent else "initializing"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
