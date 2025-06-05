import os
import asyncio
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType


class SQLDatabaseIndexer:
    """Index SQL database schema and sample data for natural language queries."""
    
    def __init__(self, memory: Memory, db_path: str):
        self.memory = memory
        self.db_path = db_path
        
    def _get_database_schema(self) -> Dict[str, Any]:
        """Extract database schema information."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        schema_info = {}
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            schema_info[table_name] = {}
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            schema_info[table_name]['columns'] = []
            for col in columns:
                schema_info[table_name]['columns'].append({
                    'name': col[1],
                    'type': col[2],
                    'not_null': col[3],
                    'default': col[4],
                    'primary_key': col[5]
                })
            
            # Get sample data (first 3 rows)
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
            sample_data = cursor.fetchall()
            schema_info[table_name]['sample_data'] = sample_data
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            schema_info[table_name]['row_count'] = row_count
        
        conn.close()
        return schema_info
    
    def _format_schema_for_indexing(self, schema_info: Dict[str, Any]) -> List[str]:
        """Format schema information for vector indexing."""
        documents = []
        
        for table_name, table_info in schema_info.items():
            # Create comprehensive table description
            doc = f"""
            Table: {table_name}
            Row Count: {table_info['row_count']}
            
            Columns:
            """
            
            for col in table_info['columns']:
                doc += f"- {col['name']} ({col['type']})"
                if col['primary_key']:
                    doc += " [PRIMARY KEY]"
                if col['not_null']:
                    doc += " [NOT NULL]"
                doc += "\n"
            
            # Add sample data context
            if table_info['sample_data']:
                doc += f"\nSample Data (first 3 rows):\n"
                column_names = [col['name'] for col in table_info['columns']]
                doc += f"Columns: {', '.join(column_names)}\n"
                
                for i, row in enumerate(table_info['sample_data'], 1):
                    doc += f"Row {i}: {row}\n"
            
            documents.append(doc)
        
        return documents
    
    async def index_database_schema(self) -> int:
        """Index database schema and sample data into vector memory."""
        try:
            print("Extracting database schema...")
            schema_info = self._get_database_schema()
            
            print("Formatting schema for indexing...")
            documents = self._format_schema_for_indexing(schema_info)
            
            total_chunks = 0
            for i, doc in enumerate(documents):
                await self.memory.add(
                    MemoryContent(
                        content=doc,
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"source": "database_schema", "chunk_index": i}
                    )
                )
                total_chunks += 1
            
            print(f"Successfully indexed {total_chunks} schema documents")
            return total_chunks
            
        except Exception as e:
            print(f"Error indexing database schema: {str(e)}")
            return 0


class SQLQueryExecutor:
    """Execute SQL queries and format results."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute SQL query and return formatted results."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Use pandas for better formatting
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            result = {
                'success': True,
                'data': df.to_dict('records'),
                'columns': df.columns.tolist(),
                'row_count': len(df),
                'formatted_table': df.to_string(index=False)
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': None,
                'columns': None,
                'row_count': 0,
                'formatted_table': None
            }


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
    print(f"Sample database created at: {db_path}")


class SQLNaturalLanguageAgent:
    """Main agent for handling natural language SQL queries."""
    
    def __init__(self, db_path: str, openai_api_key: str):
        self.db_path = db_path
        self.query_executor = SQLQueryExecutor(db_path)
        
        # Initialize vector memory
        persistence_path = os.path.join(str(Path.home()), ".chromadb_sql_agent")
        self.rag_memory = ChromaDBVectorMemory(
            config=PersistentChromaDBVectorMemoryConfig(
                collection_name="sql_database_schema",
                persistence_path=persistence_path,
                k=5,  # Return top 5 relevant schema pieces
                score_threshold=0.3,
            )
        )
        
        # Initialize OpenAI client
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4o",
            temperature=0.1,  # Lower temperature for more consistent SQL generation
            api_key=openai_api_key,
        )
        
        self.assistant = None
    
    def extract_sql_query(self, text: str) -> Optional[str]:
        """Extract SQL query from the assistant's response."""
        import re
        
        # Look for SQL queries in code blocks
        sql_patterns = [
            r'```sql\s*(.*?)\s*```',
            r'```\s*(SELECT.*?);?\s*```',
            r'```\s*(INSERT.*?);?\s*```',
            r'```\s*(UPDATE.*?);?\s*```',
            r'```\s*(DELETE.*?);?\s*```',
            r'```\s*(CREATE.*?);?\s*```',
            r'SQL Query:\s*(.*?)(?:\n|$)',
            r'Query:\s*(.*?)(?:\n|$)',
        ]
        
        for pattern in sql_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                # Clean up the query
                query = re.sub(r'\s+', ' ', query)
                if not query.endswith(';'):
                    query += ';'
                return query
        
        return None
    
    async def initialize(self):
        """Initialize the agent and index database schema if needed."""
        try:
            # Check if schema is already indexed
            import chromadb
            client = chromadb.PersistentClient(
                path=os.path.join(str(Path.home()), ".chromadb_sql_agent")
            )
            
            try:
                collection = client.get_collection("sql_database_schema")
                count = collection.count()
                print(f"Found existing schema collection with {count} documents")
                
                if count == 0:
                    raise Exception("Collection is empty")
                    
            except Exception:
                print("Indexing database schema...")
                indexer = SQLDatabaseIndexer(self.rag_memory, self.db_path)
                await indexer.index_database_schema()
            
            # Create the assistant agent with memory
            self.assistant = AssistantAgent(
                name="sql_assistant",
                model_client=self.model_client,
                memory=[self.rag_memory],
                system_message="""You are a SQL expert assistant that helps users query databases using natural language.

Your task is to:
1. Understand the user's natural language question
2. Use the database schema information from your memory to understand table structures
3. Generate appropriate SQL queries
4. Present the SQL query clearly in a code block with ```sql

When generating SQL queries:
- Always use proper SQL syntax
- Consider table relationships and foreign keys
- Use appropriate JOINs when needed
- Include LIMIT clauses for large result sets when appropriate
- Validate that column names and table names exist in the schema
- Format the SQL query clearly in a ```sql code block

IMPORTANT: Always provide the SQL query in this exact format:
```sql
YOUR_SQL_QUERY_HERE
```

The query will be automatically executed and results will be shown to the user.
"""
            )
            
        except Exception as e:
            print(f"Error initializing SQL agent: {e}")
            raise
    
    async def query(self, natural_language_question: str):
        """Process a natural language question and return SQL results."""
        if not self.assistant:
            await self.initialize()
        
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
                    
                    # Provide additional insights
                    if result['row_count'] > 10:
                        print(f"💡 Note: Showing all {result['row_count']} results. Consider adding LIMIT for large datasets.")
                else:
                    print("📭 No results found for this query.")
                    
            else:
                print("❌ Query execution failed!")
                print(f"Error: {result['error']}")
                print()
                print("💡 The assistant will learn from this error for future queries.")
        else:
            print("⚠️  Could not extract SQL query from response.")
            print("The assistant provided an explanation but no executable query.")
        
        print("=" * 60)
    
    async def close(self):
        """Close the memory connection."""
        try:
            await self.rag_memory.close()
        except Exception as e:
            print(f"Error closing memory: {e}")


async def main():
    """Main function to demonstrate the SQL natural language interface."""
    
    # Setup paths
    db_path = "sample_company.db"
    
    # Create sample database
    create_sample_database(db_path)
    
    # Initialize the SQL agent
    agent = SQLNaturalLanguageAgent(
        db_path=db_path,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    try:
        await agent.initialize()
        
        # Example queries
        questions = [
            "Show me all employees in the Engineering department",
            "What's the average salary by department?",
            "Which projects are currently in progress?",
            "Show me employees and their managers",
            "What's the total budget across all departments?"
        ]
        
        print("\n" + "="*60)
        print("SQL Natural Language Query Assistant")
        print("="*60)
        
        for question in questions:
            print(f"\n🤔 Question: {question}")
            print("-" * 50)
            await agent.query(question)
            print("\n" + "="*60)
        
        # Interactive mode
        print("\n💬 Interactive Mode - Ask your own questions (type 'quit' to exit):")
        while True:
            user_question = input("\nYour question: ").strip()
            if user_question.lower() in ['quit', 'exit', 'q']:
                break
            if user_question:
                await agent.query(user_question)
    
    except Exception as e:
        print(f"Error in main: {e}")
    
    finally:
        await agent.close()


if __name__ == "__main__":
    # Make sure you have OPENAI_API_KEY set in your environment
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    asyncio.run(main())
