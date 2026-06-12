from dotenv import load_dotenv

import Ollama
from langsmith import traceable

load_dotenv()

max_iterations = 10
modelName = 'gpt-oss:latest'

@traceable(run_type="tool")
def get_product_price(product_name: str) -> float:
    """Seach Tool to get price of a product from the catalog"""
    print(f"Searching for {product_name} in the catalog")
    prices = {"laptop": 1029.99, "mouse": 10.99, "keyboard": 20.99}
    return prices[product_name]

@traceable(run_type="tool")
def get_product_discount(category: str,price: float) -> float:
    """Seach Tool to get discount of a product from the catalog"""
    print(f"Applying discount for {category} in the catalog")
    categories_discounts = {"Gold": 20.0, "Silver": 10.0, "Bronze": 5.0}
    discount = categories_discounts.get(category, 0)
    return round(price * (1 - discount / 100), 2)


@traceable(name="LangChain Agent Loop")
def run_agent(query: str) -> str:
    tools = [get_product_price, get_product_discount]
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(f"ollama:{modelName}",temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {query}")
    print("="*100)


    messages = [
        SystemMessage(content=(
            "You are a helpful Shopping Assistant."
            "You have access to a product Catalog tool"
            "and a discount tool. \n\n"
            "STRICT RULES - you must follow these exactly: \n"
            "1. NEVER guess or assume a product price"
            "You MUST Call get_product_price tool to get the price of a product mentioned in the user's question"
            "2. Only call apply_discount tool after getting price from get_product_price tool. Pass the exact price."
            "3. NEVER Calculate discount yourself using math."
            "Alwasy use the get_product_discount tool to get the discount of a product."
            "4. If user does not specify a category, ask the user which category. Do not assume one"
            )
        ),
        HumanMessage(content=query)
        
    ]

    for iteration in range(1,max_iterations+1):
        print(f"\nIteration {iteration}")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        too_call_id = tool_call['id']

        print(f"Tool Call: {tool_name} {tool_args} {too_call_id}")


        tool_to_use = tools_dict[tool_name]
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"Observation: {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=observation, tool_call_id=too_call_id))




# Agent Loop



if __name__ == "__main__":
    print("Agent Loop")

    print()

    result = run_agent("What is the price of laptop with a Gold Category Discount?")



