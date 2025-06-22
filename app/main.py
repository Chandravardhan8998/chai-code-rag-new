from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.controllers.run_query import extract_topic, State, extract_sub_topic, run_query_with_prompt

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/run-query")
async def run_query(prompt:str):
    return await run_query_with_prompt(prompt)

# @app.post("/test-extract_topic")
# async def test_extract_topic():
#     state:State={
#   "prompt": "How to use git in SQL",
#   "response": [
#     {
#       "role": "user",
#       "content": "How to use git in SQL"
#     },
#     {
#       "role": "assistant",
#       "content": "The topic involves using git, a version control system, in the context of SQL, which relates to managing SQL code or scripts with git."
#     },
#     {
#       "role": "user",
#       "content": "How to use git in SQL"
#     },
#     {
#       "role": "assistant",
#       "content": "All sub-topics related to git in SQL are provided."
#     }
#   ],
#   "selected_topics": [
#     {
#       "name": "git-aur-git",
#       "paths": [
#         "welcome",
#         "introduction",
#         "terminology",
#         "behind-the-scenes",
#         "branches",
#         "diff-stash-tags",
#         "managing-history",
#         "github"
#       ]
#     },
#     {
#       "name": "chai-aur-sql",
#       "paths": [
#         "welcome",
#         "introduction",
#         "postgres",
#         "normalization",
#         "database-design-exercise",
#         "joins-and-keys",
#         "joins-exercise"
#       ]
#     }
#   ]
# }
#     return await extract_sub_topic(state)