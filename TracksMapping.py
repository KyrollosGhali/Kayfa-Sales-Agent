from pydantic_ai import Agent
import json
from pydantic import BaseModel
from dotenv import load_dotenv
import os
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
class TrackMapping(BaseModel):
    track_name: str
    videos: int
    hours: float
    courses: int
    price_usd: float
    source_url: str

system_prompt = """
You will receive a text containing information about list of tracks.
You task is to extract the following information from the provided text and return it in JSON format:
list[
- track_name: The name of the track.
- videos: The number of videos in the track.
- hours: The total duration of the track in hours.
- courses: The number of courses in the track.
- price_usd: The price of the track in USD.
- source_url: The URL of the source where the track information was obtained.
]
"""
courses_paid=r"data\text\kayfa_paid_educational_tracks.md"
with open(courses_paid , "r") as f:
    text = f.read()
    
print(text)
agent = Agent(
    model="groq:llama-3.1-8b-instant",
    output_type=list[TrackMapping],
    system_prompt=system_prompt,
)
result = agent.run_sync(text).output
with open("data/json/kayfa_paid_educational_tracks.json", "w") as f:
    json.dump([item.dict() for item in result], f, indent=4)

print("JSON file created successfully.")
print("Extracted data:")
for item in result:
    print(item.dict())