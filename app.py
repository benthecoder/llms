import asyncio
import logging

import streamlit as st

from llm import complete

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_async(models, prompts):
    try:
        return await complete(prompts, models=models, use_cache=True)
    except Exception as e:
        logger.error(f"Error in run_async: {e}", exc_info=True)
        raise


def display_responses(responses, column_count):
    columns = st.columns(column_count)

    # Find the maximum number of lines in any response
    max_lines = 0
    for response in responses:
        for _, answer in response["responses"].items():
            max_lines = max(max_lines, answer.count("\n") + 1)

    # Calculate the height to be used for all text areas
    uniform_height = max(100, max_lines * 40)  # 20 pixels per line as an estimate

    for response in responses:
        for i, (model, answer) in enumerate(response["responses"].items()):
            with columns[i % column_count]:
                st.text_area(
                    model,
                    value=answer,
                    height=uniform_height,
                    key=f"{model}_{i}",
                )


def main():
    st.set_page_config(
        page_title="LLMs",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("LLMs ✨")

    # User input
    user_prompt = st.text_area("Enter your prompt:", height=100)

    # Sidebar for model selection
    st.sidebar.title("Settings")
    with st.sidebar.expander("Choose Model"):
        all_models = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "replicate/mistral-7b",
            "replicate/llama-2-13b",
            "pplx/pplx-7b-online",
            "pplx/pplx-70b-online",
            "pplx/llama-2-70b-chat",
            "pplx/codellama-34b-instruct",
            "pplx/mistral-7b-instruct",
        ]
        selected_models = [model for model in all_models if st.checkbox(model, True)]

    # Column layout selection
    column_layout = st.sidebar.radio("Select no. of columns", [1, 2, 3], index=2)

    if st.button("Submit"):
        if user_prompt and selected_models:
            try:
                with st.spinner("Fetching responses..."):
                    responses = asyncio.run(run_async(selected_models, [user_prompt]))
                display_responses(responses, column_layout)
            except Exception as e:
                st.error("An error occurred while fetching responses.")
                logger.error(f"Streamlit UI Error: {e}", exc_info=True)
        elif not user_prompt:
            st.error("Please enter a prompt.")
        elif not selected_models:
            st.error("Please select at least one model.")


if __name__ == "__main__":
    main()
