import sqlite3
from typing import List, Any
from mcp.server.fastmcp import FastMCP

import mcp.types as types
from contextlib import closing
from pathlib import Path
from pydantic import AnyUrl
import json

# mcp = FastMCP("SQLiteDB")

from mcp.server.stdio import stdio_server
from mcp.server import InitializationOptions
from mcp.server.lowlevel import Server, NotificationOptions

mcp = Server("SQLiteDB")

# Change this path to point to your actual SQLite database file
DB_PATH = "sqlite_server/sql_db.db"

class SqliteDatabase:
    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path).expanduser())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self.insights: list[str] = []

    def _init_database(self):
        """Initialize connection to the SQLite database"""
        # print("Initializing database connection")
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            conn.close()

    def _synthesize_memo(self) -> str:
        """Synthesizes business insights into a formatted memo"""
        # print(f"Synthesizing memo with {len(self.insights)} insights")
        if not self.insights:
            return "No business insights have been discovered yet."

        insights = "\n".join(f"- {insight}" for insight in self.insights)

        memo = "📊 Business Intelligence Memo 📊\n\n"
        memo += "Key Insights Discovered:\n\n"
        memo += insights

        if len(self.insights) > 1:
            memo += "\nSummary:\n"
            memo += f"Analysis has revealed {len(self.insights)} key business insights that suggest opportunities for strategic optimization and growth."

        # print("Generated basic memo format")
        return memo
    
    def _get_schema(self, table_name: str) -> str:
        try:
            with open('schema.json', 'r') as f:
                data = json.load(f)
            return data[table_name]
        except FileNotFoundError:
            raise FileNotFoundError("schema.json file not found")
        except KeyError:
            raise KeyError(f"Table '{table_name}' not found in schema.json")

    def _execute_query(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a SQL query and return results as a list of dictionaries"""
        # print(f"Executing query: {query}")
        try:
            print(query)
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                with closing(conn.cursor()) as cursor:
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)

                    # if not query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER')):
                    #     conn.commit()
                    #     affected = cursor.rowcount
                    #     # print(f"Write query affected {affected} rows")
                    #     return [{"affected_rows": affected}]

                    results = [dict(row) for row in cursor.fetchall()]
                    # print(f"Read query returned {len(results)} rows")
                    return results
        except Exception as e:
            # print(f"Database error executing query: {e}")
            raise


db = SqliteDatabase(DB_PATH)

@mcp.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
            types.Tool(
                name="read_query",
                description="Execute a SELECT query on the SQLite database",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "SELECT SQL query to execute"},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="list_tables",
                description="List all tables in the SQLite database",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            types.Tool(
                name="describe_table",
                description="Get the schema information for a specific table",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "table_name": {"type": "string", "description": "Name of the table to describe"},
                    },
                    "required": ["table_name"],
                },
            ),
            types.Tool(
                name="append_insight",
                description="Add a business insight to the memo",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "insight": {"type": "string", "description": "Business insight discovered from data analysis"},
                    },
                    "required": ["insight"],
                },
            ),
        ]



@mcp.call_tool()
async def handle_tool_call(
    name: str, arguments: dict[str, Any] | None
    )-> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution requests"""
    try:
        if name == "list_tables":
            results = db._execute_query(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            return [types.TextContent(type="text", text=str(results))]

        elif name == "describe_table":
            if not arguments or "table_name" not in arguments:
                raise ValueError("Missing table_name argument")
            # results = db._execute_query(
            #     f"PRAGMA table_info({arguments['table_name']})"
            # )
            # results = db._get_schema(arguments['table_name'])
            # return [types.TextContent(type="text", text=str(results))]

            results = db._execute_query(
                    f"PRAGMA table_info({arguments['table_name']})"
                )
            return [types.TextContent(type="text", text=str(results))]

        elif name == "append_insight":
            if not arguments or "insight" not in arguments:
                raise ValueError("Missing insight argument")

            db.insights.append(arguments["insight"])
            memo = db._synthesize_memo()

            # Notify clients that the memo resource has changed
            # await mcp.request_context.session.send_resource_updated(AnyUrl("memo://insights"))

            return [types.TextContent(type="text", text="Insight added to memo")]

        if not arguments:
            raise ValueError("Missing arguments")

        if name == "read_query":
            # FIXME : update here
            if not arguments["query"].strip().upper().startswith("SELECT"):
                raise ValueError("Only SELECT queries are allowed for read_query")
            results = db._execute_query(arguments["query"])
            return [types.TextContent(type="text", text=str(results))]

        # elif name == "create_table":
        #     if not arguments["query"].strip().upper().startswith("CREATE TABLE"):
        #         raise ValueError("Only CREATE TABLE statements are allowed")
        #     db._execute_query(arguments["query"])
        #     return [types.TextContent(type="text", text="Table created successfully")]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except sqlite3.Error as e:
        return [types.TextContent(type="text", text=f"Database error: {str(e)}")]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]



async def main():
    async with stdio_server() as (read_stream, write_stream):
            print("Server running with stdio transport")
            await mcp.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="sqlite",
                    server_version="0.1.0",
                    capabilities=mcp.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

import asyncio

if __name__ == "__main__":
    # mcp.run(transport="stdio")
    # mcp.run(transport="sse")
    asyncio.run(main())


