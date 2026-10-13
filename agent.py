from langchain.agents import create_agent
from llm import create_llm
from shared_tools import lookup_interview_slots, match_resume_to_jd
from system_prompts import SYSTEM_PROMPT_LOCAL_TOOLS
from rag import reg_chain
import gradio as gr

# List tools required for screening and scheduling
tools = [lookup_interview_slots, match_resume_to_jd]

# Initialize LLM and Agent
llm = create_llm()

hr_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT_LOCAL_TOOLS
)

def run_candidate_faq(query: str) -> str:
    """Handles Candidate FAQ by directly querying the policy handbook via RAG."""
    return reg_chain(query, doc_type_filter="policy")

def run_screening(query: str) -> str:
    """Handles candidate screening by triggering the resume-to-JD match flow."""
    user_prompt = f"Screen the candidate resume against the job description. Details: {query}"
    response = hr_agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
    return response["messages"][-1].content

def run_scheduling(user_query: str) -> str:
    """Handles interview scheduling queries including single dates and multi-day date ranges."""
    response = hr_agent.invoke({"messages": [{"role": "user", "content": user_query}]})
    return response["messages"][-1].content

def deploy_agent():
    with gr.Blocks(title="HR & Hiring Onboarding Agent") as iface:
        gr.Markdown("## 🤖 HR & Hiring Onboarding Agent")
        gr.Markdown("A multi-functional hiring assistant keeping Candidate Policy FAQ, Resume Screening, and Interview Scheduling cleanly separated.")

        with gr.Tabs():
            # 1. Candidate FAQ Flow
            with gr.TabItem("📖 Candidate FAQ"):
                gr.Markdown("### Ask questions about company policies, benefits, and guidelines")
                faq_input = gr.Textbox(label="Policy Question", placeholder="e.g., What is the remote work policy or leave benefit?")
                faq_btn = gr.Button("Search Handbook Policy", variant="primary")
                faq_output = gr.Textbox(label="Policy Answer", lines=10)
                faq_btn.click(fn=run_candidate_faq, inputs=faq_input, outputs=faq_output)

            # 2. Candidate Screening Flow
            with gr.TabItem("📄 Resume Screening"):
                gr.Markdown("### Screen Candidate Resume against Job Description")
                screen_input = gr.Textbox(label="Screening Instructions", placeholder="e.g., Match the candidate resume against the JD requirements.")
                screen_btn = gr.Button("Run Resume Match", variant="primary")
                screen_output = gr.Textbox(label="Structured Screening Analysis", lines=12)
                screen_btn.click(fn=run_screening, inputs=screen_input, outputs=screen_output)

            # 3. Interview Scheduling Flow
            with gr.TabItem("📅 Interview Calendar & Scheduling"):
                gr.Markdown("### Query Full Calendar & Interview Availability")
                schedule_input = gr.Textbox(
                    label="Calendar Query / Date Range", 
                    placeholder="e.g., How many interviews are scheduled between October 11th and 17th, 2026?"
                )
                schedule_btn = gr.Button("Check Calendar", variant="primary")
                schedule_output = gr.Textbox(label="Schedule Summary", lines=10)
                schedule_btn.click(fn=run_scheduling, inputs=schedule_input, outputs=schedule_output)

    iface.launch()

if __name__ == '__main__':
    deploy_agent()