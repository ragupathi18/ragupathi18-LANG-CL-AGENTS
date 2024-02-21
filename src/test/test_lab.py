"""
This file will contain test cases for the automatic evaluation of your
solution in main/lab.py. You should not modify the code in this file. You should
also manually test your solution by running app.py.
"""

import unittest

from langchain.chat_models import AzureChatOpenAI
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

from src.main.lab import agent_executor


class TestLLMResponse(unittest.TestCase):
    """
    This test will verify that the connection to an external LLM is made. If it does not
    work, this may be because the API key is invalid, or the service may be down.
    If that is the case, this lab may not be completable.
    """
    def test_llm_sanity_check(self):
        llm = HuggingFaceEndpoint(
            endpoint_url="https://z8dvl7fzhxxcybd8.eu-west-1.aws.endpoints.huggingface.cloud",
            huggingfacehub_api_token="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",
            task="text-generation",
            model_kwargs={
                "max_new_tokens": 1024
            }
        )

        self.assertIsInstance(llm, HuggingFaceEndpoint)

    """
    This test will verify that the agent uses the appropriate tool for the task given.
    """
    def test_appropriate_tools_used_by_agent(self):

        agent_executor.max_iterations = 1

        response = agent_executor.invoke(
            {"input": "What is the length of the word Jurassic?"},
        )

        tool_used = response["intermediate_steps"][0][0].tool

        # Verifies that the get_word_tool is used when the agent is asked to find the length of a word
        self.assertEqual("get_word_length", tool_used)

        response = agent_executor.invoke(
            {"input": "What is 3 cubed?"},
        )

        tool_used = response["intermediate_steps"][0][0].tool

        # Verifies that the get_cube_of_number is used when the agent is asked to find the cube of a number
        self.assertEqual("get_cube_of_number", tool_used)

    """
    This test will verify that the agent produces the correct word length.
    """
    def test_agent_gets_length_of_word(self):

        agent_executor.max_iterations = 1

        response = agent_executor.invoke({"input": "What is the length of the word Jurassic?"})

        # have to grab output this way due to some weirdness with huggingface's behavior with langchain agents
        print(response["intermediate_steps"][0][1])

        self.assertEqual(response["intermediate_steps"][0][1], 8)

    """
    This test will verify that the agent produces the correct cube.
    """
    def test_agent_gets_cube_of_number(self):

        response = agent_executor.invoke("what is 3 cubed?",)

        self.assertEqual(response["output"], "27")
