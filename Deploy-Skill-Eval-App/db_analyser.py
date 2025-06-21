import os
import asyncio
import sqlite3
import pandas as pd
import re
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


class QuizSQLQueryExecutor:
    """Execute SQL queries on quiz database and format results."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute SQL query and return formatted results."""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return {
                'success': True,
                'data': df.to_dict('records'),
                'columns': df.columns.tolist(),
                'row_count': len(df),
                'formatted_table': df.to_string(index=False)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': None,
                'columns': None,
                'row_count': 0,
                'formatted_table': None
            }


class QuizSchemaExtractor:
    """Extract quiz database schema information."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_schema_info(self) -> str:
        """Get complete quiz database schema as a formatted string."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        schema_text = "QUIZ APPLICATION DATABASE SCHEMA:\n" + "="*60 + "\n\n"
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            schema_text += f"📊 TABLE: {table_name}\n"
            schema_text += "-" * 40 + "\n"
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            schema_text += "COLUMNS:\n"
            for col in columns:
                col_info = f"  • {col[1]} ({col[2]})"
                if col[5]:  # primary key
                    col_info += " [PRIMARY KEY]"
                if col[3]:  # not null
                    col_info += " [NOT NULL]"
                if col[4]:  # default value
                    col_info += f" [DEFAULT: {col[4]}]"
                schema_text += col_info + "\n"
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            schema_text += f"TOTAL RECORDS: {row_count}\n"
            
            # Add table-specific insights
            if table_name == 'questions':
                schema_text += "\n📝 PURPOSE: Stores generated questions for different domains and skills\n"
                # Get domain distribution
                cursor.execute("SELECT domain, COUNT(*) FROM questions GROUP BY domain ORDER BY COUNT(*) DESC;")
                domain_stats = cursor.fetchall()
                if domain_stats:
                    schema_text += "DOMAIN DISTRIBUTION:\n"
                    for domain, count in domain_stats:
                        schema_text += f"  • {domain}: {count} questions\n"
                
            elif table_name == 'answers':
                schema_text += "\n✍️  PURPOSE: Stores user answers with AI evaluation and course recommendations\n"
                # Get average scores by domain
                cursor.execute("""
                    SELECT domain, 
                           AVG(ai_score) as avg_score, 
                           COUNT(*) as total_answers,
                           COUNT(course_recommendations) as with_recommendations
                    FROM answers 
                    WHERE ai_score IS NOT NULL 
                    GROUP BY domain 
                    ORDER BY avg_score DESC;
                """)
                score_stats = cursor.fetchall()
                if score_stats:
                    schema_text += "PERFORMANCE STATISTICS:\n"
                    for domain, avg_score, total, with_recs in score_stats:
                        schema_text += f"  • {domain}: Avg Score {avg_score:.1f}/10 ({total} answers, {with_recs} with recommendations)\n"
            
            # Get sample data (first 2 rows for questions, 1 row for answers due to length)
            sample_limit = 1 if table_name == 'answers' else 2
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {sample_limit};")
            sample_data = cursor.fetchall()
            
            if sample_data:
                schema_text += f"\nSAMPLE DATA (first {sample_limit} row{'s' if sample_limit > 1 else ''}):\n"
                column_names = [col[1] for col in columns]
                
                for i, row in enumerate(sample_data, 1):
                    schema_text += f"  Row {i}:\n"
                    for j, value in enumerate(row):
                        # Handle JSON fields and long text
                        if column_names[j] in ['skills'] and isinstance(value, str):
                            try:
                                parsed_skills = json.loads(value)
                                display_value = f"[{', '.join(parsed_skills)}]"
                            except:
                                display_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                        elif column_names[j] in ['question', 'answer', 'ai_feedback', 'course_recommendations']:
                            display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                        else:
                            display_value = value
                        
                        schema_text += f"    {column_names[j]}: {display_value}\n"
                    schema_text += "\n"
            
            schema_text += "="*60 + "\n\n"
        
        # Add helpful query examples
        schema_text += "💡 COMMON QUERY PATTERNS:\n"
        schema_text += "-" * 30 + "\n"
        schema_text += "• Domain analysis: SELECT domain, COUNT(*) FROM questions GROUP BY domain\n"
        schema_text += "• Performance by domain: SELECT domain, AVG(ai_score) FROM answers GROUP BY domain\n"
        schema_text += "• Recent activity: SELECT * FROM answers WHERE submitted_at >= datetime('now', '-7 days')\n"
        schema_text += "• Skills analysis: Look for JSON parsing of skills column\n"
        schema_text += "• Course recommendations: Filter answers WHERE course_recommendations IS NOT NULL\n\n"
        
        conn.close()
        return schema_text


class QuizSQLNaturalLanguageAgent:
    """Natural language SQL agent for quiz database analysis."""
    
    def __init__(self, db_path: str, openai_api_key: str):
        self.db_path = db_path
        self.query_executor = QuizSQLQueryExecutor(db_path)
        self.schema_extractor = QuizSchemaExtractor(db_path)
        
        # Get schema once and keep in memory
        self.schema_info = self.schema_extractor.get_schema_info()
        
        # Initialize OpenAI client
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4o",
            temperature=0.1,
            api_key=openai_api_key,
        )
        
        # Create system message with schema
        system_message = f"""You are a SQL expert assistant specializing in analyzing quiz application data.

Here is the complete database schema you're working with:

{self.schema_info}

IMPORTANT DOMAIN INFORMATION:
Valid domains in this quiz system are: GenAI, Cloud, DevOps, Drone, Robotics, Data Engineering, Cybersecurity, Space Technology, Analytics, Business, Management

SPECIAL HANDLING REQUIRED:
1. The 'skills' column in both tables contains JSON arrays of skills (e.g., ["Python", "Machine Learning"])
2. When querying skills, use JSON functions like json_extract() or json_each()
3. The 'ai_feedback' and 'course_recommendations' columns contain long text responses
4. Timestamps are in SQLite datetime format

Your task is to:
1. Understand the user's natural language question about quiz data
2. Generate appropriate SQL queries based on the schema above
3. Always provide the SQL query in a ```sql code block
4. Handle JSON data properly when working with skills
5. Use appropriate date/time functions for temporal queries
6. Consider using JOINs between questions and answers tables when needed

QUERY GUIDELINES:
- For skills analysis, use: json_each(skills) to expand JSON arrays
- For date filtering, use: datetime() functions
- For text search in feedback/recommendations, use: LIKE operator
- Always include LIMIT for potentially large result sets
- Use proper GROUP BY with aggregate functions

EXAMPLE PATTERNS:
```sql
-- Skills analysis
SELECT json_extract(value, '$') as skill, COUNT(*) 
FROM questions, json_each(questions.skills) 
GROUP BY json_extract(value, '$');

-- Performance by domain
SELECT domain, AVG(ai_score) as avg_score, COUNT(*) as total_answers
FROM answers 
WHERE ai_score IS NOT NULL 
GROUP BY domain;
```

Always provide your SQL query in this exact format:
```sql
YOUR_SQL_QUERY_HERE
```

The query will be automatically executed and results will be shown to the user.
"""
        
        self.assistant = AssistantAgent(
            name="quiz_sql_assistant",
            model_client=self.model_client,
            system_message=system_message
        )
    
    def extract_sql_query(self, text: str) -> Optional[str]:
        """Extract SQL query from the assistant's response."""
        sql_patterns = [
            r'```sql\s*(.*?)\s*```',
            r'```\s*(SELECT.*?);?\s*```',
            r'```\s*(WITH.*?);?\s*```',
            r'```\s*(INSERT.*?);?\s*```',
            r'```\s*(UPDATE.*?);?\s*```',
            r'```\s*(DELETE.*?);?\s*```',
        ]
        
        for pattern in sql_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                query = re.sub(r'\s+', ' ', query)
                if not query.endswith(';'):
                    query += ';'
                return query
        
        return None
    
    async def query(self, natural_language_question: str):
        """Process a natural language question about quiz data and return SQL results."""
        print(f"🤔 Processing: {natural_language_question}")
        print("-" * 70)
        
        # Get the assistant's response
        response = await self.assistant.run(task=natural_language_question)
        
        # Print the assistant's explanation
        print("🤖 Assistant Response:")
        print(response.messages[-1].content)
        print()
        
        # Extract SQL query from response
        sql_query = self.extract_sql_query(response.messages[-1].content)
        
        if sql_query:
            print("🔍 Extracted SQL Query:")
            print(f"```sql\n{sql_query}\n```")
            print()
            
            # Execute the query
            print("⚡ Executing query...")
            result = self.query_executor.execute_query(sql_query)
            
            if result['success']:
                print("✅ Query executed successfully!")
                print(f"📊 Found {result['row_count']} rows")
                print()
                
                if result['row_count'] > 0:
                    print("📋 Results:")
                    print(result['formatted_table'])
                    print()
                    
                    if result['row_count'] > 20:
                        print(f"💡 Note: Showing all {result['row_count']} results. Consider adding LIMIT for large datasets.")
                else:
                    print("📭 No results found for this query.")
                    
            else:
                print("❌ Query execution failed!")
                print(f"🚨 Error: {result['error']}")
                print("💡 Please try rephrasing your question or check the query syntax.")
        else:
            print("⚠️  Could not extract SQL query from response.")
            print("The assistant provided an explanation but no executable query.")
        
        print("=" * 70)


def check_and_create_sample_quiz_data(db_path: str):
    """Check if quiz database exists and has data, create sample if empty."""
    
    if not os.path.exists(db_path):
        print(f"❌ Database {db_path} not found!")
        print("Please run your main.py quiz application first to create the database.")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    if 'questions' not in tables or 'answers' not in tables:
        print("❌ Required tables (questions, answers) not found in database!")
        print("Please run your main.py quiz application first to initialize the database.")
        conn.close()
        return False
    
    # Check if we have some data
    cursor.execute("SELECT COUNT(*) FROM questions;")
    question_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM answers;")
    answer_count = cursor.fetchone()[0]
    
    print(f"📊 Database Status:")
    print(f"  • Questions: {question_count}")
    print(f"  • Answers: {answer_count}")
    
    if question_count == 0:
        print("\n⚠️  No questions found in database.")
        print("💡 Consider running your quiz application and generating some questions first.")
        print("   This SQL agent works best when there's data to analyze!")
    
    conn.close()
    return True


async def main():
    """Main function to demonstrate the quiz SQL natural language interface."""
    
    # Setup
    db_path = "quiz_app.db"  # This should match your main.py database name
    
    print("🎯 Quiz Database SQL Natural Language Agent")
    print("=" * 70)
    
    # Check if database exists and has data
    if not check_and_create_sample_quiz_data(db_path):
        return
    
    # Initialize the SQL agent
    try:
        agent = QuizSQLNaturalLanguageAgent(
            db_path=db_path,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        print("Please check your OPENAI_API_KEY environment variable.")
        return
    
    try:
        # Example queries specific to quiz application
        questions = [
            "How many questions have been generated for each domain?",
            "What's the average AI score for answers in each domain?",
            "Show me the most recent 5 answers with their scores",
            "Which skills are most commonly tested across all questions?",
        ]
        
        print("\n🎯 Running example queries for quiz analysis:")
        print("=" * 70)
        
        for question in questions:
            print(f"\n❓ Question: {question}")
            await agent.query(question)
            input("\nPress Enter to continue to next question...")
        
        # Interactive mode
        print("\n💬 Interactive Mode - Ask questions about your quiz data!")
        print("Examples:")
        print("  • 'What domains are performing best?'")
        print("  • 'Show me recent low-scoring answers'")
        print("  • 'Which skills need more questions?'")
        print("  • 'How are course recommendations being used?'")
        print("\nType 'quit', 'exit', or 'q' to exit")
        print("-" * 70)
        
        while True:
            user_question = input("\n❓ Your question about quiz data: ").strip()
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Happy analyzing!")
                break
            if user_question:
                print()
                await agent.query(user_question)
    
    except Exception as e:
        print(f"❌ Error in main: {e}")


if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    # Required packages info
    print("📦 Required packages: autogen-agentchat autogen-ext openai pandas")
    print("Install with: pip install autogen-agentchat autogen-ext[openai] pandas")
    print()
    
    asyncio.run(main())
