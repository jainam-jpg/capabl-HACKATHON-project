from langchain_core.tools import tool
from rag import vector_store
from dotenv import load_dotenv
import json
import datetime

load_dotenv()

# Full Month Mock Interview Calendar (October 2026)
MOCK_CALENDAR = {
    "2026-10-01": ["09:00 AM", "11:00 AM"],
    "2026-10-02": ["02:00 PM"],
    "2026-10-05": ["10:00 AM", "01:00 PM", "04:00 PM"],
    "2026-10-06": ["11:00 AM", "03:00 PM"],
    "2026-10-08": ["09:00 AM"],
    "2026-10-12": ["10:00 AM", "02:00 PM"],
    "2026-10-13": ["09:30 AM", "11:30 AM", "03:30 PM"],
    "2026-10-14": ["01:00 PM", "04:00 PM"],
    "2026-10-15": ["10:00 AM", "02:00 PM", "04:00 PM"],
    "2026-10-16": ["11:00 AM", "03:00 PM"],
    "2026-10-17": ["09:00 AM", "01:00 PM"],
    "2026-10-20": ["10:00 AM", "11:00 AM", "02:00 PM", "05:00 PM"],
    "2026-10-21": ["03:00 PM"],
    "2026-10-23": ["09:00 AM", "01:00 PM"],
    "2026-10-26": ["10:00 AM", "02:00 PM"],
    "2026-10-28": ["11:00 AM", "04:00 PM"],
    "2026-10-30": ["09:00 AM", "12:00 PM", "03:00 PM"]
}

@tool
def lookup_interview_slots(start_date: str, end_date: str = None) -> str:
    """
    Looks up available interview time slots or counts scheduled interviews across 
    a single date or a range of dates (in YYYY-MM-DD format).
    
    Inputs:
    - start_date (str): The primary date or start of date range in YYYY-MM-DD format (e.g., '2026-10-11').
    - end_date (str, optional): The end date of range in YYYY-MM-DD format (e.g., '2026-10-17').
    
    Returns:
    Detailed listing of available interview slots and total counts for the requested timeframe.
    """
    if not end_date:
        end_date = start_date

    try:
        start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        return "Error: Dates must be provided in YYYY-MM-DD format (e.g., 2026-10-15)."

    results = {}
    total_slots = 0

    curr_dt = start_dt
    while curr_dt <= end_dt:
        date_str = curr_dt.strftime("%Y-%m-%d")
        slots = MOCK_CALENDAR.get(date_str, [])
        if slots:
            results[date_str] = slots
            total_slots += len(slots)
        else:
            results[date_str] = "No slots available"
        curr_dt += datetime.timedelta(days=1)

    summary = {
        "timeframe": f"{start_date} to {end_date}",
        "total_available_slots": total_slots,
        "daily_breakdown": results
    }
    return json.dumps(summary, indent=2)


@tool
def match_resume_to_jd() -> str:
    """
    Retrieves candidate resume chunks and job description chunks from the vector store 
    and returns a structured comparison showing matched skills and missing skills.
    """
    resume_docs = vector_store.similarity_search("skills experience", k=4, filter={"doc_type": "resume"})
    resume_text = "\n".join([doc.page_content for doc in resume_docs])
    
    jd_docs = vector_store.similarity_search("requirements responsibilities skills", k=4, filter={"doc_type": "policy"})
    jd_text = "\n".join([doc.page_content for doc in jd_docs])
    
    structured_data = {
        "candidate_resume_summary": resume_text if resume_text else "No resume chunks found.",
        "job_description_requirements": jd_text if jd_text else "No job description chunks found.",
        "instruction": "Please analyze the above resume summary against the job description requirements and provide a structured comparison detailing matched skills and missing skills."
    }
    return json.dumps(structured_data, indent=2)