from loguru import logger
import gradio as gr
from search.service import semantic_search


demo = gr.Interface(
    fn=semantic_search,
    inputs=[
        gr.components.Textbox(label="query", lines=5),
        gr.components.Textbox(label="filter", lines=5),
        gr.components.Slider(5, 20, value=10, label="k", step=1),
    ],
    outputs=gr.components.DataFrame(
        label="Semantically similar resumes", wrap=True
    ),
)


if __name__ == "__main__":
    logger.add("app.log", format="{time} {level} {message}", level="INFO")
    logger.info("Starting app...")
    demo.launch()
