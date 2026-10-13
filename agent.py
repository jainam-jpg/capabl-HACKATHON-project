from langchain.agents import create_agent
from llm import create_llm
from shared_tools import lookup_interview_slots, match_resume_to_jd
from system_prompts import SYSTEM_PROMPT_LOCAL_TOOLS
from rag import reg_chain
import gradio as gr

tools = [lookup_interview_slots, match_resume_to_jd]
llm = create_llm()

hr_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT_LOCAL_TOOLS
)

def run_candidate_faq(query: str) -> str:
    if not query.strip():
        return "Please enter a valid question about company policies or benefits."
    return reg_chain(query, doc_type_filter="policy")

def run_screening(query: str) -> str:
    if not query.strip():
        return "Please specify what you would like to evaluate regarding the resume and job description."
    user_prompt = f"Screen candidate resume against job description based on this request: {query}"
    response = hr_agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
    return response["messages"][-1].content

def run_scheduling(user_query: str) -> str:
    if not user_query.strip():
        return "Please ask a question regarding interview slots, dates, or schedule availability."
    response = hr_agent.invoke({"messages": [{"role": "user", "content": user_query}]})
    return response["messages"][-1].content

def deploy_agent():
    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue")
    
    with gr.Blocks(title="AI HR & Hiring Onboarding Assistant", theme=theme) as iface:
        gr.Markdown("# 🤖 Enterprise AI HR & Hiring Assistant")
        gr.Markdown("An intelligent multi-workflow agent featuring strict data isolation between **Company Policy FAQ**, **Dynamic Resume Screening**, and **Advanced Calendar Analytics**.")

        with gr.Tabs():
            # 1. Candidate FAQ Flow
            with gr.TabItem("📖 Candidate FAQ"):
                gr.Markdown("### Ask questions about company policies, leave rules, working hours, and benefits.")
                faq_input = gr.Textbox(label="Your Question", placeholder="e.g., What is our parental leave policy and remote work model?")
                faq_btn = gr.Button("Search Policy Handbook", variant="primary")
                faq_output = gr.Textbox(label="Policy Response", lines=8)
                faq_btn.click(fn=run_candidate_faq, inputs=faq_input, outputs=faq_output)

            # 2. Candidate Screening Flow
            with gr.TabItem("📄 Resume Screening"):
                gr.Markdown("### Ask anything regarding resume alignment, matched skills, or missing gaps against the Job Description.")
                screen_input = gr.Textbox(label="Screening Query", placeholder="e.g., What technical skills am I missing compared to the job description?")
                screen_btn = gr.Button("Analyze Candidate", variant="primary")
                screen_output = gr.Textbox(label="Screening Analysis", lines=10)
                screen_btn.click(fn=run_screening, inputs=screen_input, outputs=screen_output)

            # 3. Interview Scheduling Flow
            with gr.TabItem("📅 Interview Calendar & Analytics"):
                gr.Markdown("### Query the full October 2026 interview schedule using natural language.")
                schedule_input = gr.Textbox(
                    label="Calendar Query", 
                    placeholder="e.g., How many interviews are scheduled between October 11th and 20th? Are there back-to-back interview days?"
                )
                schedule_btn = gr.Button("Analyze Schedule", variant="primary")
                schedule_output = gr.Textbox(label="Calendar Insights", lines=10)
                schedule_btn.click(fn=run_scheduling, inputs=schedule_input, outputs=schedule_output)

    iface.launch()

if __name__ == '__main__':
    deploy_agent()