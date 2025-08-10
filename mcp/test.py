
import asyncio
test_content = 'test' #test, sqlite

if test_content == "test":

    # test server
    from test_servers.client import main
    question = "what is (3 * 2) + 1. also what is the weather in bangalore ?"

elif test_content == 'sqlite':

    # sqlite server
    from sqlite_server.client import main
    question = "what is the total number of patients?"
    # question = "show me what type of encounters are most common among patients with different races"
    # question = "list of top patients by number of encounters."
else:
    main = ""


if main: 
    # run main function in async

    response= asyncio.run(main(question))

    print("\n", "-*-"*40)
    print(f"Question: {question}\nResponse: ")
    print(response["messages"][-1].content)
    print("\n", "-*-"*40)
else:
    raise ValueError("Error: Could not import main function!")
