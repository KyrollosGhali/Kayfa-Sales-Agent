from pydantic_ai import Agent
import json
from pydantic import BaseModel
from dotenv import load_dotenv
import os
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
class CourseMapping(BaseModel):
    Course_Name: str
    Videos: int
    Hours: float
    Price_USD: float
    Instructor_Focus: str

system_prompt = """
Extract course information.

IMPORTANT:
Return ONLY valid JSON.
Do NOT use markdown.
Do NOT use tool calls.
Do NOT use XML tags.
Do NOT explain anything.

Expected format:

[
  {
    "Course_Name": "string",
    "Videos": 0,
    "Hours": 0.0,
    "Price_USD": 0.0,
    "Instructor_Focus": "string"
  }
]
"""
courses_paid=r"data\text\kayfa_paid_individual_courses.md"
with open(courses_paid , "r") as f:
    text = f.read()
    
print(text)
agent = Agent(
    model="groq:llama-3.1-8b-instant",
    output_type=list[CourseMapping],
    system_prompt=system_prompt,
    retries=5
)
result = agent.run_sync(text).output
with open("data/json/kayfa_paid_indiviual_courses.json", "w") as f:
    json.dump([item.dict() for item in result], f, indent=4)

print("JSON file created successfully.")
print("Extracted data:")
for item in result:
    print(item.dict())