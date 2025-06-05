import os
import asyncio
import sqlite3
import pandas as pd
import re
from typing import Dict, Any, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


class SimpleSQLQueryExecutor:
    """Execute SQL queries and format results."""
    
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


class SimpleSchemaExtractor:
    """Extract database schema information directly."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_schema_info(self) -> str:
        """Get complete database schema as a formatted string."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        schema_text = "DATABASE SCHEMA INFORMATION:\n" + "="*50 + "\n\n"
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            schema_text += f"📋 TABLE: {table_name}\n"
            schema_text += "-" * 30 + "\n"
            
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
                schema_text += col_info + "\n"
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            schema_text += f"ROW COUNT: {row_count}\n"
            
            # Get sample data (first 3 rows)
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
            sample_data = cursor.fetchall()
            
            if sample_data:
                schema_text += "SAMPLE DATA (first 3 rows):\n"
                column_names = [col[1] for col in columns]
                
                # Create a simple table format
                for i, row in enumerate(sample_data, 1):
                    row_data = []
                    for j, value in enumerate(row):
                        row_data.append(f"{column_names[j]}={value}")
                    schema_text += f"  Row {i}: {', '.join(row_data)}\n"
            
            schema_text += "\n" + "="*50 + "\n\n"
        
        conn.close()
        return schema_text


class SimpleSQLNaturalLanguageAgent:
    """Simplified SQL agent without ChromaDB dependency."""
    
    def __init__(self, db_path: str, openai_api_key: str):
        self.db_path = db_path
        self.query_executor = SimpleSQLQueryExecutor(db_path)
        self.schema_extractor = SimpleSchemaExtractor(db_path)
        
        # Get schema once and keep in memory
        self.schema_info = self.schema_extractor.get_schema_info()
        
        # Initialize OpenAI client
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4o",
            temperature=0.1,
            api_key=openai_api_key,
        )
        
        # Create system message with schema
        system_message = f"""You are a SQL expert assistant that helps users query databases using natural language.

Here is the complete database schema you're working with:

{self.schema_info}

Your task is to:
1. Understand the user's natural language question
2. Generate appropriate SQL queries based on the schema above
3. Always provide the SQL query in a ```sql code block

When generating SQL queries:
- Use the exact table and column names from the schema above
- Consider table relationships and use JOINs when needed
- Use proper SQL syntax for SQLite
- Include LIMIT clauses for potentially large result sets
- Validate that all referenced tables and columns exist in the schema

IMPORTANT: Always provide your SQL query in this exact format:
```sql
YOUR_SQL_QUERY_HERE
```

The query will be automatically executed and results will be shown to the user.
"""
        
        self.assistant = AssistantAgent(
            name="simple_sql_assistant",
            model_client=self.model_client,
            system_message=system_message
        )
    
    def extract_sql_query(self, text: str) -> Optional[str]:
        """Extract SQL query from the assistant's response."""
        sql_patterns = [
            r'```sql\s*(.*?)\s*```',
            r'```\s*(SELECT.*?);?\s*```',
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
        """Process a natural language question and return SQL results."""
        print(f"🤔 Processing: {natural_language_question}")
        print("-" * 60)
        
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
                    
                    if result['row_count'] > 10:
                        print(f"💡 Note: Showing all {result['row_count']} results.")
                else:
                    print("📭 No results found for this query.")
                    
            else:
                print("❌ Query execution failed!")
                print(f"Error: {result['error']}")
                print("💡 Please try rephrasing your question.")
        else:
            print("⚠️  Could not extract SQL query from response.")
            print("The assistant provided an explanation but no executable query.")
        
        print("=" * 60)


def create_sample_database(db_path: str):
    """Create a sample database for demonstration."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create employees table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary REAL,
            hire_date DATE,
            manager_id INTEGER
        )
    ''')
    
    # Create departments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            budget REAL,
            location TEXT
        )
    ''')
    
    # Create projects table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department_id INTEGER,
            start_date DATE,
            end_date DATE,
            status TEXT,
            FOREIGN KEY (department_id) REFERENCES departments (id)
        )
    ''')
    
    # Insert sample data
    employees_data = [
        (1, 'John Doe', 'Engineering', 85000, '2020-01-15', None),
        (2, 'Jane Smith', 'Engineering', 90000, '2019-03-20', 1),
        (3, 'Mike Johnson', 'Sales', 70000, '2021-06-10', None),
        (4, 'Sarah Wilson', 'Marketing', 65000, '2020-08-05', None),
        (5, 'Tom Brown', 'Engineering', 75000, '2022-01-12', 2),
        (6, 'Lisa Davis', 'Sales', 68000, '2021-09-15', 3),
        (7, 'Bob Martinez', 'Marketing', 62000, '2022-03-01', 4),
        (8, 'Alice Cooper', 'HR', 72000, '2020-11-20', None)
    ]
    
    departments_data = [
        (1, 'Engineering', 500000, 'Building A'),
        (2, 'Sales', 300000, 'Building B'),
        (3, 'Marketing', 200000, 'Building B'),
        (4, 'HR', 150000, 'Building C')
    ]
    
    projects_data = [
        (1, 'Web Platform Redesign', 1, '2023-01-01', '2023-06-30', 'Completed'),
        (2, 'Mobile App Development', 1, '2023-03-01', '2023-12-31', 'In Progress'),
        (3, 'Q4 Sales Campaign', 2, '2023-10-01', '2023-12-31', 'In Progress'),
        (4, 'Brand Refresh', 3, '2023-02-01', '2023-08-31', 'Completed'),
        (5, 'Employee Onboarding System', 4, '2023-05-01', '2023-11-30', 'In Progress')
    ]
    
    cursor.executemany('INSERT OR REPLACE INTO employees VALUES (?, ?, ?, ?, ?, ?)', employees_data)
    cursor.executemany('INSERT OR REPLACE INTO departments VALUES (?, ?, ?, ?)', departments_data)
    cursor.executemany('INSERT OR REPLACE INTO projects VALUES (?, ?, ?, ?, ?, ?)', projects_data)
    
    conn.commit()
    conn.close()
    print(f"✅ Sample database created at: {db_path}")


async def main():
    """Main function to demonstrate the simple SQL natural language interface."""
    
    # Setup
    db_path = "simple_company.db"
    
    print("🚀 Setting up Simple SQL Natural Language Agent")
    print("=" * 60)
    
    # Create sample database
    create_sample_database(db_path)
    
    # Initialize the SQL agent (no ChromaDB needed!)
    agent = SimpleSQLNaturalLanguageAgent(
        db_path=db_path,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    try:
        # Example queries
        questions = [
            "How many employees do we have?",
            "What's the average salary by department?",
            "Show me all employees in Engineering",
            "Which projects are currently in progress?",
            "Who are the highest paid employees?"
        ]
        
        print("\n📋 Running example queries:")
        print("=" * 60)
        
        for question in questions:
            print(f"\n❓ Question: {question}")
            await agent.query(question)
            input("\nPress Enter to continue to next question...")
        
        # Interactive mode
        print("\n💬 Interactive Mode - Ask your own questions!")
        print("Type 'quit', 'exit', or 'q' to exit")
        print("-" * 60)
        
        while True:
            user_question = input("\n❓ Your question: ").strip()
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
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
    
    # Required packages
    print("📦 Required packages: autogen-agentchat autogen-ext openai pandas")
    print("💡 Install with: pip install autogen-agentchat autogen-ext[openai] pandas")
    print()
    
    asyncio.run(main())
