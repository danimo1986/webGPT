import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain.tools import DuckDuckGoSearchTool

# Install required packages
# !pip install langchain==0.0.149
# !pip install openai==0.27.8
# !pip install duckduckgo-search==3.8.3

# Create Streamlit app
def main():
    # Page title
    st.title("OpenAI Chat App")

    # Sidebar for API key input
    user_api_key = st.sidebar.text_input(
        label="OpenAI API key",
        placeholder="Paste your OpenAI API key here",
        type="password")

    # Check if the API key is provided
    if not user_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        return

    # Set OpenAI API key
    os.environ["OPENAI_API_KEY"] = user_api_key

    # Create OpenAI model and tools
    llm = OpenAI(model_name="text-davinci-003", temperature=0.2)
    search = DuckDuckGoSearchTool()

    tools = [
        Tool(
            name="duckduckgo-search",
            func=search.run,
            description="Useful for when you need to search for the latest information on the web",
        )
    ]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )

    # User input text box
    user_input = st.text_area("User Input:", "")

    if st.button("Submit"):
        if user_input.strip() != "":
            # Add user input to the conversation history
            agent.memory.append_message("user", user_input)

            # Get the response from the chatbot
            response = agent.generate_output()

            # Display the chat history
            st.write("Chat History:")
            for message in agent.memory.get_memory()["messages"]:
                st.write(f"{message['role']}: {message['content']}")

            # Display the chatbot's response
            st.write("Chatbot Response:")
            st.write(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
